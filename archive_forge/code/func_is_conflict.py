from neutronclient.common import exceptions
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.v2_0 import client as nc
from oslo_config import cfg
from oslo_utils import uuidutils
from heat.common import exception
from heat.common.i18n import _
from heat.engine.clients import client_plugin
from heat.engine.clients import os as os_client
def is_conflict(self, ex):
    bad_conflicts = (exceptions.OverQuotaClient,)
    return isinstance(ex, exceptions.Conflict) and (not isinstance(ex, bad_conflicts))