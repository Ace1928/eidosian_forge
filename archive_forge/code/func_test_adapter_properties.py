from keystoneauth1 import session
from oslo_utils import uuidutils
from novaclient import api_versions
from novaclient.tests.unit import utils
from novaclient.v2 import client
def test_adapter_properties(self):
    user_agent = uuidutils.generate_uuid(dashed=False)
    endpoint_override = uuidutils.generate_uuid(dashed=False)
    s = session.Session()
    c = client.Client(session=s, api_version=api_versions.APIVersion('2.0'), user_agent=user_agent, endpoint_override=endpoint_override, direct_use=False)
    self.assertEqual(user_agent, c.client.user_agent)
    self.assertEqual(endpoint_override, c.client.endpoint_override)