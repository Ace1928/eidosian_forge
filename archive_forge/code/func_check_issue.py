from boto.mws.connection import MWSConnection
from boto.mws.response import (ResponseFactory, ResponseElement, Element,
from tests.unit import AWSMockServiceTestCase
from boto.compat import filter, map
from tests.compat import unittest
def check_issue(self, klass, text):
    action = klass.__name__[:-len('Result')]
    factory = ResponseFactory(scopes=[{klass.__name__: klass}])
    parser = factory(action, connection=self.service_connection)
    return self.service_connection._parse_response(parser, 'text/xml', text)