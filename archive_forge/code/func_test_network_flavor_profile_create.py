from openstackclient.tests.functional.network.v2 import common
def test_network_flavor_profile_create(self):
    json_output = self.openstack('network flavor profile create ' + '--description ' + self.DESCRIPTION + ' ' + '--enable --metainfo ' + self.METAINFO, parse_output=True)
    ID = json_output.get('id')
    self.assertIsNotNone(ID)
    self.assertTrue(json_output.get('enabled'))
    self.assertEqual('fakedescription', json_output.get('description'))
    self.assertEqual('Extrainfo', json_output.get('meta_info'))
    raw_output = self.openstack('network flavor profile delete ' + ID)
    self.assertOutput('', raw_output)