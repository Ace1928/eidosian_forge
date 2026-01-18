from openstack.network.v2 import bgpvpn as _bgpvpn
from openstack.network.v2 import (
from openstack.network.v2 import bgpvpn_port_association as _bgpvpn_port_assoc
from openstack.network.v2 import (
from openstack.network.v2 import network as _network
from openstack.network.v2 import port as _port
from openstack.network.v2 import router as _router
from openstack.network.v2 import subnet as _subnet
from openstack.tests.functional import base
def test_find_bgpvpn(self):
    sot = self.operator_cloud.network.find_bgpvpn(self.BGPVPN.name)
    self.assertEqual(list(self.ROUTE_TARGETS), sot.route_targets)
    self.assertEqual(list(self.IMPORT_TARGETS), sot.import_targets)
    self.assertEqual(self.TYPE, sot.type)