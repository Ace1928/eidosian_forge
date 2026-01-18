from unittest import mock
import fixtures
from oslotest import base as test_base
import webob.dec
import webob.exc
from oslo_middleware import catch_errors
def test_filter_tokens_from_log(self):
    logger = self.useFixture(fixtures.FakeLogger(nuke_handlers=False))

    @webob.dec.wsgify
    def application(req):
        raise Exception()
    app = catch_errors.CatchErrors(application)
    req = webob.Request.blank('/test', text='test data', method='POST', headers={'X-Auth-Token': 'secret1', 'X-Service-Token': 'secret2', 'X-Other-Token': 'secret3'})
    res = req.get_response(app)
    self.assertEqual(500, res.status_int)
    output = logger.output
    self.assertIn('X-Auth-Token: *****', output)
    self.assertIn('X-Service-Token: *****', output)
    self.assertIn('X-Other-Token: *****', output)
    self.assertIn('test data', output)