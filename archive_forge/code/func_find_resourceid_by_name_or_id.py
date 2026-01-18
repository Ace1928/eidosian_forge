from neutronclient.common import exceptions
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.v2_0 import client as nc
from oslo_config import cfg
from oslo_utils import uuidutils
from heat.common import exception
from heat.common.i18n import _
from heat.engine.clients import client_plugin
from heat.engine.clients import os as os_client
def find_resourceid_by_name_or_id(self, resource, name_or_id, cmd_resource=None):
    """Find a resource ID given either a name or an ID.

        The `resource` argument should be one of the constants defined in
        RES_TYPES.
        """
    cmd_resource = cmd_resource or self._res_cmdres_mapping.get(resource)
    return self._find_resource_id(self.context.tenant_id, resource, name_or_id, cmd_resource)