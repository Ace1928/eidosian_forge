import re
import sys
import warnings
def lint_app(*args, **kw):
    assert_(len(args) == 2, 'Two arguments required')
    assert_(not kw, 'No keyword arguments allowed')
    environ, start_response = args
    check_environ(environ)
    start_response_started = []

    def start_response_wrapper(*args, **kw):
        assert_(len(args) == 2 or len(args) == 3, 'Invalid number of arguments: %s' % (args,))
        assert_(not kw, 'No keyword arguments allowed')
        status = args[0]
        headers = args[1]
        if len(args) == 3:
            exc_info = args[2]
        else:
            exc_info = None
        check_status(status)
        check_headers(headers)
        check_content_type(status, headers)
        check_exc_info(exc_info)
        start_response_started.append(None)
        return WriteWrapper(start_response(*args))
    environ['wsgi.input'] = InputWrapper(environ['wsgi.input'])
    environ['wsgi.errors'] = ErrorWrapper(environ['wsgi.errors'])
    iterator = application(environ, start_response_wrapper)
    assert_(iterator is not None and iterator != False, 'The application must return an iterator, if only an empty list')
    check_iterator(iterator)
    return IteratorWrapper(iterator, start_response_started)