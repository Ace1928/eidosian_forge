import openstack.cloud
from openstack.tests.unit import base
def test_ironic_noauth_none_auth_type(self):
    """Test noauth selection for Ironic in OpenStackCloud

        The new way of doing this is with the keystoneauth none plugin.
        """
    self.cloud_noauth = openstack.connect(auth_type='none', baremetal_endpoint_override='https://baremetal.example.com')
    self.cloud_noauth.list_machines()
    self.assert_calls()