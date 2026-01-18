import fixtures
import hashlib
import uuid
import warnings
from oslotest import base as test_base
from oslo_context import context
from oslo_context import fixture
def test_auth_token_info_removed_logging_values(self):
    ctx = TestContext(auth_token_info={'auth_token': 'topsecret'})
    d = ctx.get_logging_values()
    self.assertNotIn('auth_token_info', d)