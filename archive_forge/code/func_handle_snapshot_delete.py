import copy
import ipaddress
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import uuidutils
from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine.clients import progress
from heat.engine import constraints
from heat.engine import properties
from heat.engine.resources.openstack.neutron import port as neutron_port
from heat.engine.resources.openstack.neutron import subnet
from heat.engine.resources.openstack.nova import server_network_mixin
from heat.engine.resources import scheduler_hints as sh
from heat.engine.resources import server_base
from heat.engine import support
from heat.engine import translation
from heat.rpc import api as rpc_api
def handle_snapshot_delete(self, state):
    if state[1] != self.FAILED and self.resource_id:
        image_id = self.client().servers.create_image(self.resource_id, self.physical_resource_name())
        return progress.ServerDeleteProgress(self.resource_id, image_id, False)
    return self._delete()