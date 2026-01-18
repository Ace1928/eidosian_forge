from openstack.network.v2 import bgpvpn as _bgpvpn
from openstack.network.v2 import (
from openstack.network.v2 import bgpvpn_port_association as _bgpvpn_port_assoc
from openstack.network.v2 import (
from openstack.network.v2 import network as _network
from openstack.network.v2 import port as _port
from openstack.network.v2 import router as _router
from openstack.network.v2 import subnet as _subnet
from openstack.tests.functional import base
def test_get_bgpvpn_port_association(self):
    sot = self.operator_cloud.network.get_bgpvpn_port_association(self.BGPVPN.id, self.PORT_ASSOC.id)
    self.assertEqual(self.PORT.id, sot.port_id)