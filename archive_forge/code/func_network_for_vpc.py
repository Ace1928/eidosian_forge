from heat.common import exception
from heat.common.i18n import _
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine.resources.openstack.neutron import neutron
@staticmethod
def network_for_vpc(client, network_id):
    return client.show_network(network_id)['network']