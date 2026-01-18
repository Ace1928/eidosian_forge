from neutronclient.common import exceptions
from neutronclient.neutron import v2_0 as neutronV20
from neutronclient.v2_0 import client as nc
from oslo_config import cfg
from oslo_utils import uuidutils
from heat.common import exception
from heat.common.i18n import _
from heat.engine.clients import client_plugin
from heat.engine.clients import os as os_client
def get_qos_policy_id(self, policy):
    """Returns the id of QoS policy.

        Args:
        policy: ID or name of the policy.
        """
    return self.find_resourceid_by_name_or_id('policy', policy)