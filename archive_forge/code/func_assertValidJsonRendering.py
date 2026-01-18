import uuid
import fixtures
from oslo_config import fixture as config_fixture
from oslo_log import log
from oslo_serialization import jsonutils
import keystone.conf
from keystone import exception
from keystone.server.flask.request_processing.middleware import auth_context
from keystone.tests import unit
def assertValidJsonRendering(self, e):
    resp = auth_context.render_exception(e)
    self.assertEqual(e.code, resp.status_int)
    self.assertEqual('%s %s' % (e.code, e.title), resp.status)
    j = jsonutils.loads(resp.body)
    self.assertIsNotNone(j.get('error'))
    self.assertIsNotNone(j['error'].get('code'))
    self.assertIsNotNone(j['error'].get('title'))
    self.assertIsNotNone(j['error'].get('message'))
    self.assertNotIn('\n', j['error']['message'])
    self.assertNotIn('  ', j['error']['message'])
    self.assertIs(type(j['error']['code']), int)