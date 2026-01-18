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
def stub_auth_data(self, **kwargs):
    token = self.get_auth_data(**kwargs)
    self.user_id = token.user_id
    try:
        self.project_id = token.project_id
    except AttributeError:
        self.project_id = token.tenant_id
    self.stub_auth(json=token)