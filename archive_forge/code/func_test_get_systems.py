import uuid
from keystoneauth1 import fixture
from keystoneclient.tests.unit.v3 import utils
from keystoneclient.v3 import auth
def test_get_systems(self):
    body = {'system': [{'all': True}]}
    self.stub_url('GET', ['auth', 'system'], json=body)
    systems = self.client.auth.systems()
    system = systems[0]
    self.assertEqual(1, len(systems))
    self.assertIsInstance(system, auth.System)
    self.assertTrue(system.all)