import fixtures
import hashlib
import uuid
import warnings
from oslotest import base as test_base
from oslo_context import context
from oslo_context import fixture
def test_from_environ_strip_roles(self):
    environ = {'HTTP_X_ROLES': ' abc\t,\ndef\n,ghi\n\n', 'HTTP_X_SERVICE_ROLES': ' jkl\t,\nmno\n,pqr\n\n'}
    ctx = context.RequestContext.from_environ(environ=environ)
    self.assertEqual(['abc', 'def', 'ghi'], ctx.roles)
    self.assertEqual(['jkl', 'mno', 'pqr'], ctx.service_roles)