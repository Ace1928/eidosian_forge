import fixtures
import hashlib
import uuid
import warnings
from oslotest import base as test_base
from oslo_context import context
from oslo_context import fixture
def test_auth_token_info_removed_redacted_context(self):
    userid = 'foo'
    ctx = TestContext(auth_token_info={'auth_token': 'topsecret'}, service_token='1234567', auth_token='8901234', user_id=userid)
    safe_ctxt = ctx.redacted_copy()
    self.assertIsNone(safe_ctxt.auth_token_info)
    self.assertIsNone(safe_ctxt.service_token)
    self.assertIsNone(safe_ctxt.auth_token)
    self.assertEqual(userid, safe_ctxt.user_id)
    self.assertNotEqual(ctx, safe_ctxt)