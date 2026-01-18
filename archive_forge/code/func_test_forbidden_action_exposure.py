import uuid
import fixtures
from oslo_config import fixture as config_fixture
from oslo_log import log
from oslo_serialization import jsonutils
import keystone.conf
from keystone import exception
from keystone.server.flask.request_processing.middleware import auth_context
from keystone.tests import unit
def test_forbidden_action_exposure(self):
    self.config_fixture.config(debug=False)
    risky_info = uuid.uuid4().hex
    action = uuid.uuid4().hex
    e = exception.ForbiddenAction(message=risky_info, action=action)
    self.assertValidJsonRendering(e)
    self.assertNotIn(risky_info, str(e))
    self.assertIn(action, str(e))
    self.assertNotIn(exception.SecurityError.amendment, str(e))
    e = exception.ForbiddenAction(action=action)
    self.assertValidJsonRendering(e)
    self.assertIn(action, str(e))
    self.assertNotIn(exception.SecurityError.amendment, str(e))