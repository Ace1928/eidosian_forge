import json
from webtest import TestApp
import pecan
from pecan.middleware.errordocument import ErrorDocumentMiddleware
from pecan.middleware.recursive import RecursiveMiddleware
from pecan.tests import PecanTestCase
def four_oh_four_app(environ, start_response):
    if environ['PATH_INFO'].startswith('/error'):
        code = environ['PATH_INFO'].split('/')[2].encode('utf-8')
        start_response('200 OK', [('Content-type', 'text/plain')])
        body = b'Error: %s' % code
        if environ['QUERY_STRING']:
            body += b'\nQS: %s' % environ['QUERY_STRING'].encode('utf-8')
        return [body]
    start_response('404 Not Found', [('Content-type', 'text/plain')])
    return []