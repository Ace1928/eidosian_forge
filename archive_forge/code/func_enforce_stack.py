from oslo_config import cfg
from oslo_log import log as logging
from oslo_policy import opts
from oslo_policy import policy
from oslo_utils import excutils
from heat.common import exception
from heat.common.i18n import _
from heat import policies
def enforce_stack(self, stack, scope=None, target=None, is_registered_policy=False):
    for res_type in stack.defn.all_resource_types():
        self.enforce(stack.context, res_type, scope=scope, target=target, is_registered_policy=is_registered_policy)