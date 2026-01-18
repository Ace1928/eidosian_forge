from openstackclient.tests.functional.identity.v2 import common
def test_ec2_credentials_delete(self):
    access_key = self._create_dummy_ec2_credentials(add_clean_up=False)
    raw_output = self.openstack('ec2 credentials delete %s' % access_key)
    self.assertEqual(0, len(raw_output))