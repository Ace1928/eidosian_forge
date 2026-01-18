from neutronclient.common import exceptions
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.v2_0 import client as nc
from oslo_config import cfg
from oslo_utils import uuidutils
from heat.common import exception
from heat.common.i18n import _
from heat.engine.clients import client_plugin
from heat.engine.clients import os as os_client
def show_ext_resource(self, resource, resource_id):
    """Returns specific ext resource record."""
    path = self._resolve_resource_path(resource)
    return self.client().show_ext(path + '/%s', resource_id).get(resource)