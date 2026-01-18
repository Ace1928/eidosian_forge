import fixtures
import hashlib
import uuid
import warnings
from oslotest import base as test_base
from oslo_context import context
from oslo_context import fixture
def test_from_dict_unknown_keys(self):
    dct = {'auth_token': 'token1', 'user': 'user1', 'read_only': True, 'roles': 'role1,role2,role3', 'color': 'red', 'unknown': ''}
    ctx = context.RequestContext.from_dict(dct)
    self.assertEqual('token1', ctx.auth_token)
    self.assertEqual('user1', ctx.user_id)
    self.assertIsNone(ctx.project_id)
    self.assertFalse(ctx.is_admin)
    self.assertTrue(ctx.read_only)
    self.assertRaises(KeyError, lambda: ctx.__dict__['color'])