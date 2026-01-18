from unittest import mock
import uuid
from openstack.network.v2 import agent as _agent
from openstack.network.v2 import bgp_peer as _bgp_peer
from openstack.network.v2 import bgp_speaker as _bgp_speaker
from neutronclient.tests.unit.osc.v2 import fakes
@staticmethod
def create_bgp_peers(attrs=None, count=1):
    """Create one or multiple fake bgp peers."""
    bgp_peers = []
    for i in range(count):
        bgp_peer = FakeBgpPeer.create_one_bgp_peer(attrs)
        bgp_peers.append(bgp_peer)
    return bgp_peers