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
def stack_task(self, action, reverse=False, post_func=None, aggregate_exceptions=False, pre_completion_func=None, notify=None):
    """A task to perform an action on the stack.

        All of the resources are traversed in forward or reverse dependency
        order.

        :param action: action that should be executed with stack resources
        :param reverse: define if action on the resources need to be executed
                        in reverse dependency order
        :param post_func: function that need to be executed after
                          action complete on the stack
        :param aggregate_exceptions: define if exceptions should be aggregated
        :param pre_completion_func: function that need to be executed right
                                    before action completion; uses stack,
                                    action, status and reason as input
                                    parameters
        """
    try:
        lifecycle_plugin_utils.do_pre_ops(self.context, self, None, action)
    except Exception as e:
        self.state_set(action, self.FAILED, e.args[0] if e.args else 'Failed stack pre-ops: %s' % str(e))
        if callable(post_func):
            post_func()
        if notify is not None:
            assert self.defer_state_persist()
        return
    self.state_set(action, self.IN_PROGRESS, 'Stack %s started' % action)
    if notify is not None:
        notify.signal()
    stack_status = self.COMPLETE
    reason = 'Stack %s completed successfully' % action
    action_method = action.lower()
    handle_kwargs = getattr(self, '_%s_kwargs' % action_method, lambda x: {})

    @functools.wraps(getattr(resource.Resource, action_method))
    def resource_action(r):
        handle = getattr(r, action_method)
        yield from handle(**handle_kwargs(r))
        if action == self.CREATE:
            stk_defn.update_resource_data(self.defn, r.name, r.node_data())

    def get_error_wait_time(resource):
        return resource.cancel_grace_period()
    action_task = scheduler.DependencyTaskGroup(self.dependencies, resource_action, reverse, error_wait_time=get_error_wait_time, aggregate_exceptions=aggregate_exceptions)
    try:
        yield from action_task()
    except scheduler.Timeout:
        stack_status = self.FAILED
        reason = '%s timed out' % action.title()
    except Exception as ex:
        stack_status = self.FAILED
        reason = 'Resource %s failed: %s' % (action, str(ex))
    if pre_completion_func:
        pre_completion_func(self, action, stack_status, reason)
    self.state_set(action, stack_status, reason)
    if callable(post_func):
        post_func()
    lifecycle_plugin_utils.do_post_ops(self.context, self, None, action, self.status == self.FAILED)