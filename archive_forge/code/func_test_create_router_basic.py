import ipaddress
from openstack import exceptions
from openstack.tests.functional import base
def test_create_router_basic(self):
    net1_name = self.network_prefix + '_net1'
    net1 = self.operator_cloud.create_network(name=net1_name, external=True)
    router_name = self.router_prefix + '_create_basic'
    router = self.operator_cloud.create_router(name=router_name, admin_state_up=True, ext_gateway_net_id=net1['id'])
    for field in EXPECTED_TOPLEVEL_FIELDS:
        self.assertIn(field, router)
    ext_gw_info = router['external_gateway_info']
    for field in EXPECTED_GW_INFO_FIELDS:
        self.assertIn(field, ext_gw_info)
    self.assertEqual(router_name, router['name'])
    self.assertEqual('ACTIVE', router['status'])
    self.assertEqual(net1['id'], ext_gw_info['network_id'])
    self.assertTrue(ext_gw_info['enable_snat'])