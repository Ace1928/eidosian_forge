import uuid
from keystoneauth1 import access
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneauth1 import session
from keystoneauth1.tests.unit import utils
def assertDiscoveryFailure(self, **kwargs):
    plugin = self.new_plugin(**kwargs)
    self.assertRaises(exceptions.DiscoveryFailure, plugin.get_auth_ref, self.session)