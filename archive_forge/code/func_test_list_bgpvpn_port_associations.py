from openstack.network.v2 import bgpvpn as _bgpvpn
from openstack.network.v2 import (
from openstack.network.v2 import bgpvpn_port_association as _bgpvpn_port_assoc
from openstack.network.v2 import (
from openstack.network.v2 import network as _network
from openstack.network.v2 import port as _port
from openstack.network.v2 import router as _router
from openstack.network.v2 import subnet as _subnet
from openstack.tests.functional import base
def test_list_bgpvpn_port_associations(self):
    port_assoc_ids = [port_assoc.id for port_assoc in self.operator_cloud.network.bgpvpn_port_associations(self.BGPVPN.id)]
    self.assertIn(self.PORT_ASSOC.id, port_assoc_ids)