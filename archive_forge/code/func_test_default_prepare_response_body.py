import unittest
from wsme import WSRoot
import wsme.protocol
import wsme.rest.protocol
from wsme.root import default_prepare_response_body
from webob import Request
def test_default_prepare_response_body(self):
    default_prepare_response_body(None, [b'a']) == b'a'
    default_prepare_response_body(None, [b'a', b'b']) == b'a\nb'
    default_prepare_response_body(None, ['a']) == 'a'
    default_prepare_response_body(None, ['a', 'b']) == 'a\nb'