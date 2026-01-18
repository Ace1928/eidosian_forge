import unittest
from wsme import WSRoot
from wsme.protocol import getprotocol, CallContext, Protocol
import wsme.protocol
def test_getprotocol():
    try:
        getprotocol('invalid')
        assert False, 'ValueError was not raised'
    except ValueError:
        pass