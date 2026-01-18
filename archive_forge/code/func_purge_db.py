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
def purge_db(self):
    """Cleanup database after stack has completed/failed.

        1. Delete the resources from DB.
        2. If the stack failed, update the current_traversal to empty string
           so that the resource workers bail out.
        3. Delete previous raw template if stack completes successfully.
        4. Deletes all sync points. They are no longer needed after stack
           has completed/failed.
        5. Delete the stack if the action is DELETE.
        """
    resource_objects.Resource.purge_deleted(self.context, self.id)
    exp_trvsl = self.current_traversal
    if self.status == self.FAILED:
        self.current_traversal = ''
    prev_tmpl_id = None
    if self.prev_raw_template_id is not None and self.status != self.FAILED:
        prev_tmpl_id = self.prev_raw_template_id
        self.prev_raw_template_id = None
    stack_id = self.store(exp_trvsl=exp_trvsl)
    if stack_id is None:
        LOG.warning('Failed to store stack %(name)s with traversal ID %(trvsl_id)s, aborting stack purge', {'name': self.name, 'trvsl_id': self.current_traversal})
        return
    if prev_tmpl_id is not None:
        raw_template_object.RawTemplate.delete(self.context, prev_tmpl_id)
    sync_point.delete_all(self.context, self.id, exp_trvsl)
    if (self.action, self.status) == (self.DELETE, self.COMPLETE):
        if not self.owner_id:
            status, reason = self._delete_credentials(self.status, self.status_reason, False)
            if status == self.FAILED:
                self.state_set(self.action, status, reason)
                return
        try:
            stack_object.Stack.delete(self.context, self.id)
        except exception.NotFound:
            pass