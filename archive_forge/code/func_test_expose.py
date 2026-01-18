import unittest
import webtest
from wsme import WSRoot, expose, validate
from wsme.rest import scan_api
from wsme import types
from wsme import exc
import wsme.api as wsme_api
import wsme.types
from wsme.tests.test_protocols import DummyProtocol
def test_expose(self):

    class MyWS(WSRoot):

        @expose(int)
        def getint(self):
            return 1
    assert MyWS.getint._wsme_definition.return_type == int