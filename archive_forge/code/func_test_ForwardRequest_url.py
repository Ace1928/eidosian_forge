from webtest import TestApp
from pecan.middleware.recursive import (RecursiveMiddleware,
from pecan.tests import PecanTestCase
def test_ForwardRequest_url(self):

    class TestForwardRequestMiddleware(Middleware):

        def __call__(self, environ, start_response):
            if environ['PATH_INFO'] != '/not_found':
                return self.app(environ, start_response)
            raise ForwardRequestException(self.url)
    forward(TestForwardRequestMiddleware(error_docs_app))