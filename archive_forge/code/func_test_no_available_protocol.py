import unittest
import webtest
from wsme import WSRoot, expose, validate
from wsme.rest import scan_api
from wsme import types
from wsme import exc
import wsme.api as wsme_api
import wsme.types
from wsme.tests.test_protocols import DummyProtocol
def test_no_available_protocol(self):
    r = WSRoot()
    app = webtest.TestApp(r.wsgiapp())
    res = app.get('/', expect_errors=True)
    print(res.status_int)
    assert res.status_int == 406
    print(res.body)
    assert res.body.find(b'None of the following protocols can handle this request') != -1