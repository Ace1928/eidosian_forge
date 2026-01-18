import fixtures
import hashlib
import uuid
import warnings
from oslotest import base as test_base
from oslo_context import context
from oslo_context import fixture
def test_from_environ_no_roles(self):
    ctx = context.RequestContext.from_environ(environ={})
    self.assertEqual([], ctx.roles)
    ctx = context.RequestContext.from_environ(environ={'HTTP_X_ROLES': ''})
    self.assertEqual([], ctx.roles)