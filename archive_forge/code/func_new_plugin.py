import uuid
from keystoneauth1 import access
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneauth1 import session
from keystoneauth1.tests.unit import utils
def new_plugin(self, **kwargs):
    kwargs.setdefault('auth_url', self.TEST_URL)
    return self.PLUGIN_CLASS(**kwargs)