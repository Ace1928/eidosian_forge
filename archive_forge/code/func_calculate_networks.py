import itertools
import eventlet
from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import netutils
import tenacity
from heat.common import exception
from heat.common.i18n import _
from heat.engine import resource
from heat.engine.resources.openstack.neutron import port as neutron_port
def calculate_networks(self, old_nets, new_nets, ifaces, security_groups=None):
    new_str_net = self._str_network(new_nets)
    if new_str_net:
        return self._calculate_using_str_network(ifaces, new_str_net, security_groups)
    else:
        return self._calculate_using_list_networks(old_nets, new_nets, ifaces, security_groups)