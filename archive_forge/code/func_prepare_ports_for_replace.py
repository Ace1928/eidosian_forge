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
def prepare_ports_for_replace(self):
    server = None
    with self.client_plugin().ignore_not_found:
        server = self.client().servers.get(self.resource_id)
    if server and server.status != 'ERROR':
        self.detach_ports(self)
    else:
        self._delete_internal_ports()