from webtest import TestApp
from pecan.middleware.recursive import (RecursiveMiddleware,
from pecan.tests import PecanTestCase
def test_ForwardRequestException(self):

    class TestForwardRequestExceptionMiddleware(Middleware):

        def __call__(self, environ, start_response):
            if environ['PATH_INFO'] != '/not_found':
                return self.app(environ, start_response)
            raise ForwardRequestException(path_info=self.url)
    forward(TestForwardRequestExceptionMiddleware(error_docs_app))