from openstack.network.v2 import bgpvpn
from openstack.network.v2 import bgpvpn_network_association
from openstack.network.v2 import bgpvpn_port_association
from openstack.network.v2 import bgpvpn_router_association
from openstack.network.v2 import network
from openstack.network.v2 import port
from openstack.network.v2 import router
from openstack.tests.unit import base
def test_create_bgpvpn_network_association(self):
    test_bpgvpn = bgpvpn.BgpVpn(**EXAMPLE)
    test_net = network.Network(**{'name': 'foo_net', 'id': NET_ID})
    sot = bgpvpn_network_association.BgpVpnNetworkAssociation(bgpvn_id=test_bpgvpn.id, network_id=test_net.id)
    self.assertEqual(test_net.id, sot.network_id)
    self.assertEqual(test_bpgvpn.id, sot.bgpvn_id)