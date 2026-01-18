import sys
import socket
from xmlrpc.client import (
import cherrypy
from cherrypy import _cptools
from cherrypy.test import helper
def testXmlRpc(self):
    scheme = self.scheme
    if scheme == 'https':
        url = 'https://%s:%s/xmlrpc/' % (self.interface(), self.PORT)
        proxy = ServerProxy(url, transport=HTTPSTransport())
    else:
        url = 'http://%s:%s/xmlrpc/' % (self.interface(), self.PORT)
        proxy = ServerProxy(url)
    self.getPage('/xmlrpc/foo')
    self.assertBody('Hello world!')
    self.assertEqual(proxy.return_single_item_list(), [42])
    self.assertNotEqual(proxy.return_single_item_list(), 'one bazillion')
    self.assertEqual(proxy.return_string(), 'here is a string')
    self.assertEqual(proxy.return_tuple(), list(('here', 'is', 1, 'tuple')))
    self.assertEqual(proxy.return_dict(), {'a': 1, 'c': 3, 'b': 2})
    self.assertEqual(proxy.return_composite(), [{'a': 1, 'z': 26}, 'hi', ['welcome', 'friend']])
    self.assertEqual(proxy.return_int(), 42)
    self.assertEqual(proxy.return_float(), 3.14)
    self.assertEqual(proxy.return_datetime(), DateTime((2003, 10, 7, 8, 1, 0, 1, 280, -1)))
    self.assertEqual(proxy.return_boolean(), True)
    self.assertEqual(proxy.test_argument_passing(22), 22 * 2)
    try:
        proxy.test_argument_passing({})
    except Exception:
        x = sys.exc_info()[1]
        self.assertEqual(x.__class__, Fault)
        self.assertEqual(x.faultString, "unsupported operand type(s) for *: 'dict' and 'int'")
    else:
        self.fail('Expected xmlrpclib.Fault')
    try:
        proxy.non_method()
    except Exception:
        x = sys.exc_info()[1]
        self.assertEqual(x.__class__, Fault)
        self.assertEqual(x.faultString, 'method "non_method" is not supported')
    else:
        self.fail('Expected xmlrpclib.Fault')
    try:
        proxy.test_returning_Fault()
    except Exception:
        x = sys.exc_info()[1]
        self.assertEqual(x.__class__, Fault)
        self.assertEqual(x.faultString, 'custom Fault response')
    else:
        self.fail('Expected xmlrpclib.Fault')