from keystoneclient.generic import client
from keystoneclient.tests.unit.v2_0 import utils
def test_get_version_local(self):
    self.stub_url('GET', base_url='http://localhost:35357/', json=self.TEST_RESPONSE_DICT)
    with self.deprecations.expect_deprecations_here():
        cs = client.Client()
    versions = cs.discover()
    self.assertIsInstance(versions, dict)
    self.assertIn('message', versions)
    self.assertIn('v2.0', versions)
    self.assertEqual(versions['v2.0']['url'], self.TEST_RESPONSE_DICT['versions']['values'][0]['links'][0]['href'])