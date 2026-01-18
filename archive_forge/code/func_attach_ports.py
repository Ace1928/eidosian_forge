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
def attach_ports(self, server):
    prev_server_id = server.resource_id
    for port in self.get_all_ports(server):
        self.client_plugin().interface_attach(prev_server_id, port['id'])
        try:
            if self.client_plugin().check_interface_attach(prev_server_id, port['id']):
                LOG.info('Attach interface %(port)s successful to server %(server)s', {'port': port['id'], 'server': prev_server_id})
        except tenacity.RetryError:
            raise exception.InterfaceAttachFailed(port=port['id'], server=prev_server_id)