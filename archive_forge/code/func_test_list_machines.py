import uuid
from testscenarios import load_tests_apply_scenarios as load_tests  # noqa
from openstack import exceptions
from openstack.network.v2 import port as _port
from openstack.tests import fakes
from openstack.tests.unit import base
def test_list_machines(self):
    fake_baremetal_two = fakes.make_fake_machine('two', str(uuid.uuid4()))
    self.register_uris([dict(method='GET', uri=self.get_mock_url(resource='nodes'), json={'nodes': [self.fake_baremetal_node, fake_baremetal_two]})])
    machines = self.cloud.list_machines()
    self.assertEqual(2, len(machines))
    self.assertSubdict(self.fake_baremetal_node, machines[0])
    self.assertSubdict(fake_baremetal_two, machines[1])
    self.assert_calls()