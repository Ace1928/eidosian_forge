from openstack.network.v2 import bgpvpn
from openstack.network.v2 import bgpvpn_network_association
from openstack.network.v2 import bgpvpn_port_association
from openstack.network.v2 import bgpvpn_router_association
from openstack.network.v2 import network
from openstack.network.v2 import port
from openstack.network.v2 import router
from openstack.tests.unit import base
def test_create_bgpvpn_port_association(self):
    test_bpgvpn = bgpvpn.BgpVpn(**EXAMPLE)
    test_port = port.Port(**{'name': 'foo_port', 'id': PORT_ID, 'network_id': NET_ID})
    sot = bgpvpn_port_association.BgpVpnPortAssociation(bgpvn_id=test_bpgvpn.id, port_id=test_port.id)
    self.assertEqual(test_port.id, sot.port_id)
    self.assertEqual(test_bpgvpn.id, sot.bgpvn_id)