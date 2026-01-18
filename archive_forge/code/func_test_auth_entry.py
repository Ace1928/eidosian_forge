import base64
import os
import tempfile
from oslo_config import cfg
import webob
from oslo_middleware import basic_auth as auth
from oslotest import base as test_base
def test_auth_entry(self):
    entry_pass = 'myName:$2y$05$lE3eGtyj41jZwrzS87KTqe6.JETVCWBkc32C63UP2aYrGoYOEpbJm'
    entry_fail = 'foo:bar'
    self.assertEqual({'HTTP_X_USER': 'myName', 'HTTP_X_USER_NAME': 'myName'}, auth.auth_entry(entry_pass, b'myPassword'))
    ex = self.assertRaises(webob.exc.HTTPBadRequest, auth.auth_entry, entry_fail, b'bar')
    self.assertEqual('Only bcrypt digested passwords are supported for foo', str(ex))
    self.assertRaises(webob.exc.HTTPUnauthorized, auth.auth_entry, entry_pass, b'bar')