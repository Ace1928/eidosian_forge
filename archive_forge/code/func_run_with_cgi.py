import os
import sys
def run_with_cgi(application):
    environ = dict(os.environ.items())
    environ['wsgi.input'] = sys.stdin
    environ['wsgi.errors'] = sys.stderr
    environ['wsgi.version'] = (1, 0)
    environ['wsgi.multithread'] = False
    environ['wsgi.multiprocess'] = True
    environ['wsgi.run_once'] = True
    if environ.get('HTTPS', 'off') in ('on', '1'):
        environ['wsgi.url_scheme'] = 'https'
    else:
        environ['wsgi.url_scheme'] = 'http'
    headers_set = []
    headers_sent = []

    def write(data):
        if not headers_set:
            raise AssertionError('write() before start_response()')
        elif not headers_sent:
            status, response_headers = headers_sent[:] = headers_set
            line = 'Status: %s\r\n' % status
            line = line.encode('utf-8')
            stdout.write(line)
            for header in response_headers:
                line = '%s: %s\r\n' % header
                line = line.encode('utf-8')
                stdout.write(line)
            stdout.write(b'\r\n')
        stdout.write(data)
        stdout.flush()

    def start_response(status, response_headers, exc_info=None):
        if exc_info:
            try:
                if headers_sent:
                    raise exc_info
            finally:
                exc_info = None
        elif headers_set:
            raise AssertionError('Headers already set!')
        headers_set[:] = [status, response_headers]
        return write
    result = application(environ, start_response)
    try:
        for data in result:
            if data:
                write(data)
        if not headers_sent:
            write('')
    finally:
        if hasattr(result, 'close'):
            result.close()