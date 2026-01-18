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
@profiler.trace('Stack.converge_stack', hide_args=False)
@reset_state_on_error
def converge_stack(self, template, action=UPDATE, new_stack=None, pre_converge=None):
    """Update the stack template and trigger convergence for resources."""
    if action not in [self.CREATE, self.ADOPT]:
        self.prev_raw_template_id = getattr(self.t, 'id', None)
    self.defn = self.defn.clone_with_new_template(template, self.identifier(), clear_resource_data=True)
    self.reset_dependencies()
    self._resources = None
    if action != self.CREATE:
        self.updated_time = oslo_timeutils.utcnow()
    if new_stack is not None:
        self.disable_rollback = new_stack.disable_rollback
        self.timeout_mins = new_stack.timeout_mins
        self.converge = new_stack.converge
        self.defn = new_stack.defn
        self._set_param_stackid()
        self.tags = new_stack.tags
    self.action = action
    self.status = self.IN_PROGRESS
    self.status_reason = 'Stack %s started' % self.action
    previous_traversal = self.current_traversal
    self.current_traversal = uuidutils.generate_uuid()
    stack_id = self.store(exp_trvsl=previous_traversal)
    if stack_id is None:
        LOG.warning('Failed to store stack %(name)s with traversal ID %(trvsl_id)s, aborting stack %(action)s', {'name': self.name, 'trvsl_id': previous_traversal, 'action': self.action})
        return
    self._send_notification_and_add_event()
    if previous_traversal:
        sync_point.delete_all(self.context, self.id, previous_traversal)
    self.thread_group_mgr.start(self.id, self._converge_create_or_update, pre_converge=pre_converge)