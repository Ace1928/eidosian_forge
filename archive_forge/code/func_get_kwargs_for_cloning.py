import collections
import contextlib
import copy
import eventlet
import functools
import re
import warnings
from oslo_config import cfg
from oslo_log import log as logging
from oslo_utils import excutils
from oslo_utils import timeutils as oslo_timeutils
from oslo_utils import uuidutils
from osprofiler import profiler
from heat.common import context as common_context
from heat.common import environment_format as env_fmt
from heat.common import exception
from heat.common.i18n import _
from heat.common import identifier
from heat.common import lifecycle_plugin_utils
from heat.engine import api
from heat.engine import dependencies
from heat.engine import environment
from heat.engine import event
from heat.engine.notification import stack as notification
from heat.engine import parameter_groups as param_groups
from heat.engine import parent_rsrc
from heat.engine import resource
from heat.engine import resources
from heat.engine import scheduler
from heat.engine import status
from heat.engine import stk_defn
from heat.engine import sync_point
from heat.engine import template as tmpl
from heat.engine import update
from heat.objects import raw_template as raw_template_object
from heat.objects import resource as resource_objects
from heat.objects import snapshot as snapshot_object
from heat.objects import stack as stack_object
from heat.objects import stack_tag as stack_tag_object
from heat.objects import user_creds as ucreds_object
from heat.rpc import api as rpc_api
from heat.rpc import worker_client as rpc_worker_client
def get_kwargs_for_cloning(self, keep_status=False, only_db=False, keep_tags=False):
    """Get common kwargs for calling Stack() for cloning.

        The point of this method is to reduce the number of places that we
        need to update when a kwarg to Stack.__init__() is modified. It
        is otherwise easy to forget an option and cause some unexpected
        error if this option is lost.

        Note:

        - This doesn't return the args(name, template) but only the kwargs.
        - We often want to start 'fresh' so don't want to maintain the old
          status, action and status_reason.
        - We sometimes only want the DB attributes.
        """
    stack = {'owner_id': self.owner_id, 'username': self.username, 'disable_rollback': self.disable_rollback, 'stack_user_project_id': self.stack_user_project_id, 'user_creds_id': self.user_creds_id, 'nested_depth': self.nested_depth, 'convergence': self.convergence, 'current_traversal': self.current_traversal, 'prev_raw_template_id': self.prev_raw_template_id, 'current_deps': self.current_deps}
    if keep_status:
        stack.update({'action': self.action, 'status': self.status, 'status_reason': str(self.status_reason)})
    if only_db:
        stack['parent_resource_name'] = self.parent_resource_name
        stack['tenant'] = self.tenant_id
        stack['timeout'] = self.timeout_mins
    else:
        stack['parent_resource'] = self.parent_resource_name
        stack['tenant_id'] = self.tenant_id
        stack['timeout_mins'] = self.timeout_mins
        stack['strict_validate'] = self.strict_validate
        if keep_tags:
            stack['tags'] = self.tags
    return stack