from collections import abc
from oslo_config import cfg
from oslo_log import log as logging
from oslo_policy import opts
from oslo_policy import policy
from glance.common import exception
from glance.domain import proxy
from glance import policies
def check_is_admin(self, context):
    """Check if the given context is associated with an admin role,
           as defined via the 'context_is_admin' RBAC rule.

           :param context: Glance request context
           :returns: A non-False value if context role is admin.
        """
    return self.check(context, 'context_is_admin', context.to_dict())