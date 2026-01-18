import datetime
import fixtures
import uuid
import freezegun
from oslo_config import fixture as config_fixture
from oslo_log import log
from keystone.common import fernet_utils
from keystone.common import utils as common_utils
import keystone.conf
from keystone.credential.providers import fernet as credential_fernet
from keystone import exception
from keystone.server.flask import application
from keystone.tests import unit
from keystone.tests.unit import ksfixtures
from keystone.tests.unit import utils
def test_debug_message_logged_when_loading_fernet_token_keys(self):
    self.useFixture(ksfixtures.KeyRepository(self.config_fixture, 'fernet_tokens', CONF.fernet_tokens.max_active_keys))
    logging_fixture = self.useFixture(fixtures.FakeLogger(level=log.DEBUG))
    fernet_utilities = fernet_utils.FernetUtils(CONF.fernet_tokens.key_repository, CONF.fernet_tokens.max_active_keys, 'fernet_tokens')
    fernet_utilities.load_keys()
    expected_debug_message = 'Loaded 2 Fernet keys from %(dir)s, but `[fernet_tokens] max_active_keys = %(max)d`; perhaps there have not been enough key rotations to reach `max_active_keys` yet?' % {'dir': CONF.fernet_tokens.key_repository, 'max': CONF.fernet_tokens.max_active_keys}
    self.assertIn(expected_debug_message, logging_fixture.output)