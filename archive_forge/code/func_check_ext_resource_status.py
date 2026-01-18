from neutronclient.common import exceptions
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.v2_0 import client as nc
from oslo_config import cfg
from oslo_utils import uuidutils
from heat.common import exception
from heat.common.i18n import _
from heat.engine.clients import client_plugin
from heat.engine.clients import os as os_client
def check_ext_resource_status(self, resource, resource_id):
    ext_resource = self.show_ext_resource(resource, resource_id)
    status = ext_resource['status']
    if status == 'ERROR':
        raise exception.ResourceInError(resource_status=status)
    return status == 'ACTIVE'