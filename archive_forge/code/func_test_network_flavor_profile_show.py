from openstackclient.tests.functional.network.v2 import common
def test_network_flavor_profile_show(self):
    json_output_1 = self.openstack('network flavor profile create ' + '--description ' + self.DESCRIPTION + ' ' + '--enable ' + '--metainfo ' + self.METAINFO, parse_output=True)
    ID = json_output_1.get('id')
    self.assertIsNotNone(ID)
    json_output = self.openstack('network flavor profile show ' + ID, parse_output=True)
    self.assertEqual(ID, json_output['id'])
    self.assertTrue(json_output['enabled'])
    self.assertEqual('fakedescription', json_output['description'])
    self.assertEqual('Extrainfo', json_output['meta_info'])
    raw_output = self.openstack('network flavor profile delete ' + ID)
    self.assertOutput('', raw_output)