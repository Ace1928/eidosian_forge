import ipaddress
from openstack import exceptions
from openstack.tests.functional import base
def test_list_router_interfaces(self):
    router = self._create_and_verify_advanced_router(external_cidr=u'10.5.5.0/24')
    net_name = self.network_prefix + '_intnet1'
    sub_name = self.subnet_prefix + '_intsub1'
    net = self.operator_cloud.create_network(name=net_name)
    sub = self.operator_cloud.create_subnet(net['id'], '10.6.6.0/24', subnet_name=sub_name, gateway_ip='10.6.6.1')
    iface = self.operator_cloud.add_router_interface(router, subnet_id=sub['id'])
    all_ifaces = self.operator_cloud.list_router_interfaces(router)
    int_ifaces = self.operator_cloud.list_router_interfaces(router, interface_type='internal')
    ext_ifaces = self.operator_cloud.list_router_interfaces(router, interface_type='external')
    self.assertIsNone(self.operator_cloud.remove_router_interface(router, subnet_id=sub['id']))
    self.assertIsNotNone(iface)
    self.assertEqual(2, len(all_ifaces))
    self.assertEqual(1, len(int_ifaces))
    self.assertEqual(1, len(ext_ifaces))
    ext_fixed_ips = router['external_gateway_info']['external_fixed_ips']
    self.assertEqual(ext_fixed_ips[0]['subnet_id'], ext_ifaces[0]['fixed_ips'][0]['subnet_id'])
    self.assertEqual(sub['id'], int_ifaces[0]['fixed_ips'][0]['subnet_id'])