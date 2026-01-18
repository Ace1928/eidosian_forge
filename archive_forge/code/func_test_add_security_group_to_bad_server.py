import copy
import openstack.cloud
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_add_security_group_to_bad_server(self):
    fake_server = fakes.make_fake_server('1234', 'server-name', 'ACTIVE')
    self.register_uris([self.get_nova_discovery_mock_dict(), dict(method='GET', uri='{endpoint}/servers/detail'.format(endpoint=fakes.COMPUTE_ENDPOINT), json={'servers': [fake_server]})])
    ret = self.cloud.add_server_security_groups('unknown-server-name', 'nova-sec-group')
    self.assertFalse(ret)
    self.assert_calls()