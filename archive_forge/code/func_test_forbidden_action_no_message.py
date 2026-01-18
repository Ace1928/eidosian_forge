import uuid
import fixtures
from oslo_config import fixture as config_fixture
from oslo_log import log
from oslo_serialization import jsonutils
import keystone.conf
from keystone import exception
from keystone.server.flask.request_processing.middleware import auth_context
from keystone.tests import unit
def test_forbidden_action_no_message(self):
    action = uuid.uuid4().hex
    self.config_fixture.config(debug=False)
    e = exception.ForbiddenAction(action=action)
    exposed_message = str(e)
    self.assertIn(action, exposed_message)
    self.assertNotIn(exception.SecurityError.amendment, str(e))
    self.config_fixture.config(debug=True)
    e = exception.ForbiddenAction(action=action)
    self.assertEqual(exposed_message, str(e))