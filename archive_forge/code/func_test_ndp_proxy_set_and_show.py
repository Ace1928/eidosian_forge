from openstackclient.tests.functional.network.v2 import common
def test_ndp_proxy_set_and_show(self):
    ndp_proxies = {'name': self.getUniqueString(), 'router_id': self.ROT_ID, 'port_id': self.INT_PORT_ID, 'address': self.INT_PORT_ADDRESS}
    description = 'balala'
    self._create_ndp_proxies([ndp_proxies])
    ndp_proxy_id = self.created_ndp_proxies[0]['id']
    output = self.openstack('router ndp proxy set --description %s %s' % (description, ndp_proxy_id))
    self.assertEqual('', output)
    json_output = self.openstack('router ndp proxy show ' + ndp_proxy_id, parse_output=True)
    self.assertEqual(ndp_proxies['name'], json_output['name'])
    self.assertEqual(ndp_proxies['router_id'], json_output['router_id'])
    self.assertEqual(ndp_proxies['port_id'], json_output['port_id'])
    self.assertEqual(ndp_proxies['address'], json_output['ip_address'])
    self.assertEqual(description, json_output['description'])