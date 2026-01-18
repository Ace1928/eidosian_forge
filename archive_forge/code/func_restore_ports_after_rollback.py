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
def restore_ports_after_rollback(self, convergence):
    if convergence:
        prev_server = self
        rsrc, rsrc_owning_stack, stack = resource.Resource.load(prev_server.context, prev_server.replaced_by, prev_server.stack.current_traversal, True, prev_server.stack.defn._resource_data)
        existing_server = rsrc
    else:
        backup_stack = self.stack._backup_stack()
        prev_server = backup_stack.resources.get(self.name)
        existing_server = self
    if existing_server.resource_id is not None:
        try:
            while True:
                active = self.client_plugin()._check_active(existing_server.resource_id)
                if active:
                    break
                eventlet.sleep(1)
        except exception.ResourceInError:
            pass
        self.store_external_ports()
        self.detach_ports(existing_server)
    self.attach_ports(prev_server)