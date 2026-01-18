import base64
import datetime
import hashlib
import os
from unittest import mock
import uuid
import fixtures
from oslo_log import log
from oslo_utils import timeutils
from keystone import auth
from keystone.common import fernet_utils
from keystone.common import provider_api
from keystone.common import utils
import keystone.conf
from keystone import exception
from keystone.federation import constants as federation_constants
from keystone.models import token_model
from keystone.tests import unit
from keystone.tests.unit import default_fixtures
from keystone.tests.unit import ksfixtures
from keystone.tests.unit.ksfixtures import database
from keystone.token import provider
from keystone.token.providers import fernet
from keystone.token import token_formatters
def test_no_warning_when_token_does_not_exceed_max_token_size(self):
    self.config_fixture.config(max_token_size=300)
    self.logging = self.useFixture(fixtures.FakeLogger(level=log.INFO))
    token = token_model.TokenModel()
    token.user_id = '0123456789abcdef0123456789abcdef0123456789abcdef'
    token.project_id = '0123456789abcdef0123456789abcdef0123456789abcdef'
    token.expires_at = utils.isotime(provider.default_expire_time(), subsecond=True)
    token.methods = ['password']
    token.audit_id = provider.random_urlsafe_str()
    token_id, issued_at = self.provider.generate_id_and_issued_at(token)
    expected_output = f'Fernet token created with length of {len(token_id)} characters, which exceeds 255 characters'
    self.assertNotIn(expected_output, self.logging.output)