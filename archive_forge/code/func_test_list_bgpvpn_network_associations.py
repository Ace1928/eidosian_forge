from openstack.network.v2 import bgpvpn as _bgpvpn
from openstack.network.v2 import (
from openstack.network.v2 import bgpvpn_port_association as _bgpvpn_port_assoc
from openstack.network.v2 import (
from openstack.network.v2 import network as _network
from openstack.network.v2 import port as _port
from openstack.network.v2 import router as _router
from openstack.network.v2 import subnet as _subnet
from openstack.tests.functional import base
def test_list_bgpvpn_network_associations(self):
    net_assoc_ids = [net_assoc.id for net_assoc in self.operator_cloud.network.bgpvpn_network_associations(self.BGPVPN.id)]
    self.assertIn(self.NET_ASSOC.id, net_assoc_ids)