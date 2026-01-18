from oslo_log import log as logging
from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources.openstack.neutron import neutron
from heat.engine.resources.openstack.neutron import port
from heat.engine.resources.openstack.neutron import router
from heat.engine import support
from heat.engine import translation
def port_on_subnet(resource, subnet):
    if not resource.has_interface('OS::Neutron::Port'):
        return False
    fixed_ips = resource.properties.get(port.Port.FIXED_IPS) or []
    for fixed_ip in fixed_ips:
        port_subnet = fixed_ip.get(port.Port.FIXED_IP_SUBNET) or fixed_ip.get(port.Port.FIXED_IP_SUBNET_ID)
        return subnet == port_subnet
    return False