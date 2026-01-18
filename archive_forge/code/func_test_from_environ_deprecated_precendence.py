import fixtures
import hashlib
import uuid
import warnings
from oslotest import base as test_base
from oslo_context import context
from oslo_context import fixture
def test_from_environ_deprecated_precendence(self):
    old = uuid.uuid4().hex
    new = uuid.uuid4().hex
    override = uuid.uuid4().hex
    environ = {'HTTP_X_USER': old, 'HTTP_X_USER_ID': new}
    ctx = context.RequestContext.from_environ(environ=environ)
    self.assertEqual(new, ctx.user)
    ctx = context.RequestContext.from_environ(environ=environ, user=override)
    self.assertEqual(override, ctx.user)
    environ = {'HTTP_X_TENANT': old, 'HTTP_X_PROJECT_ID': new}
    ctx = context.RequestContext.from_environ(environ=environ)
    self.assertEqual(new, ctx.project_id)
    ctx = context.RequestContext.from_environ(environ=environ, project_id=override)
    self.assertEqual(override, ctx.project_id)
    environ = {'HTTP_X_TENANT_NAME': old, 'HTTP_X_PROJECT_NAME': new}
    ctx = context.RequestContext.from_environ(environ=environ)
    self.assertEqual(new, ctx.project_name)