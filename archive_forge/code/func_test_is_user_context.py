import fixtures
import hashlib
import uuid
import warnings
from oslotest import base as test_base
from oslo_context import context
from oslo_context import fixture
def test_is_user_context(self):
    self.assertFalse(context.is_user_context(None))
    ctx = context.RequestContext(is_admin=True)
    self.assertFalse(context.is_user_context(ctx))
    ctx = context.RequestContext(is_admin=False)
    self.assertTrue(context.is_user_context(ctx))
    self.assertFalse(context.is_user_context('non context object'))