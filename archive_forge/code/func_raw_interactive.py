import io
import sys
import warnings
from traceback import print_exception
from io import StringIO
from urllib.parse import unquote, urlsplit
from paste.request import get_cookies, parse_querystring, parse_formvars
from paste.request import construct_url, path_info_split, path_info_pop
from paste.response import HeaderDict, has_header, header_value, remove_header
from paste.response import error_body_response, error_response, error_response_app
def raw_interactive(application, path='', raise_on_wsgi_error=False, **environ):
    """
    Runs the application in a fake environment.
    """
    assert 'path_info' not in environ, 'argument list changed'
    if raise_on_wsgi_error:
        errors = ErrorRaiser()
    else:
        errors = io.StringIO()
    basic_environ = {'REQUEST_METHOD': 'GET', 'SCRIPT_NAME': '', 'PATH_INFO': '', 'SERVER_NAME': 'localhost', 'SERVER_PORT': '80', 'SERVER_PROTOCOL': 'HTTP/1.0', 'wsgi.version': (1, 0), 'wsgi.url_scheme': 'http', 'wsgi.input': io.BytesIO(), 'wsgi.errors': errors, 'wsgi.multithread': False, 'wsgi.multiprocess': False, 'wsgi.run_once': False}
    if path:
        _, _, path_info, query, fragment = urlsplit(str(path))
        path_info = unquote(path_info)
        path_info, query = (str(path_info), str(query))
        basic_environ['PATH_INFO'] = path_info
        if query:
            basic_environ['QUERY_STRING'] = query
    for name, value in environ.items():
        name = name.replace('__', '.')
        basic_environ[name] = value
    if 'SERVER_NAME' in basic_environ and 'HTTP_HOST' not in basic_environ:
        basic_environ['HTTP_HOST'] = basic_environ['SERVER_NAME']
    istream = basic_environ['wsgi.input']
    if isinstance(istream, bytes):
        basic_environ['wsgi.input'] = io.BytesIO(istream)
        basic_environ['CONTENT_LENGTH'] = len(istream)
    data = {}
    output = []
    headers_set = []
    headers_sent = []

    def start_response(status, headers, exc_info=None):
        if exc_info:
            try:
                if headers_sent:
                    raise exc_info
            finally:
                exc_info = None
        elif headers_set:
            raise AssertionError('Headers already set and no exc_info!')
        headers_set.append(True)
        data['status'] = status
        data['headers'] = headers
        return output.append
    app_iter = application(basic_environ, start_response)
    try:
        try:
            for s in app_iter:
                if not isinstance(s, bytes):
                    raise ValueError('The app_iter response can only contain bytes (not unicode); got: %r' % s)
                headers_sent.append(True)
                if not headers_set:
                    raise AssertionError('Content sent w/o headers!')
                output.append(s)
        except TypeError as e:
            e.args = (e.args[0] + ' iterable: %r' % app_iter,) + e.args[1:]
            raise
    finally:
        if hasattr(app_iter, 'close'):
            app_iter.close()
    return (data['status'], data['headers'], b''.join(output), errors.getvalue())