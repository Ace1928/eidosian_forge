from unittest import mock
import uuid
from openstack.network.v2 import agent as _agent
from openstack.network.v2 import bgp_peer as _bgp_peer
from openstack.network.v2 import bgp_speaker as _bgp_speaker
from neutronclient.tests.unit.osc.v2 import fakes
@staticmethod
def create_dragents(attrs=None, count=1):
    """Create one or multiple fake dynamic routing agents."""
    agents = []
    for i in range(count):
        agent = FakeDRAgent.create_one_dragent(attrs)
        agents.append(agent)
    return agents