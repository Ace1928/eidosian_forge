import uuid
import fixtures
from oslo_config import fixture as config_fixture
from oslo_log import log
from oslo_serialization import jsonutils
import keystone.conf
from keystone import exception
from keystone.server.flask.request_processing.middleware import auth_context
from keystone.tests import unit
def test_nested_translation_of_SecurityErrors(self):
    e = self.CustomSecurityError(place='code')
    'Admiral found this in the log: %s' % e
    self.assertNotIn('programmer error', self.warning_log.output)