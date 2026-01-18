import abc
import collections
import urllib
import uuid
from keystoneauth1 import _utils
from keystoneauth1 import access
from keystoneauth1 import adapter
from keystoneauth1 import discover
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneauth1 import identity
from keystoneauth1 import plugin
from keystoneauth1 import session
from keystoneauth1.tests.unit import utils
def create_auth_plugin(self, **kwargs):
    kwargs.setdefault('auth_url', self.TEST_URL)
    kwargs.setdefault('username', self.TEST_USER)
    kwargs.setdefault('password', self.TEST_PASS)
    try:
        kwargs.setdefault('tenant_id', kwargs.pop('project_id'))
    except KeyError:
        pass
    try:
        kwargs.setdefault('tenant_name', kwargs.pop('project_name'))
    except KeyError:
        pass
    return identity.V2Password(**kwargs)