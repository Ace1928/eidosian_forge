import collections
import copy
import datetime
import functools
import itertools
import os
import pydoc
import signal
import socket
import sys
import eventlet
from oslo_config import cfg
from oslo_context import context as oslo_context
from oslo_log import log as logging
import oslo_messaging as messaging
from oslo_serialization import jsonutils
from oslo_service import service
from oslo_service import threadgroup
from oslo_utils import timeutils
from oslo_utils import uuidutils
from osprofiler import profiler
import webob
from heat.common import context
from heat.common import environment_format as env_fmt
from heat.common import environment_util as env_util
from heat.common import exception
from heat.common.i18n import _
from heat.common import identifier
from heat.common import messaging as rpc_messaging
from heat.common import policy
from heat.common import service_utils
from heat.engine import api
from heat.engine import attributes
from heat.engine.cfn import template as cfntemplate
from heat.engine import clients
from heat.engine import environment
from heat.engine.hot import functions as hot_functions
from heat.engine import parameter_groups
from heat.engine import properties
from heat.engine import resources
from heat.engine import service_software_config
from heat.engine import stack as parser
from heat.engine import stack_lock
from heat.engine import stk_defn
from heat.engine import support
from heat.engine import template as templatem
from heat.engine import template_files
from heat.engine import update
from heat.engine import worker
from heat.objects import event as event_object
from heat.objects import resource as resource_objects
from heat.objects import service as service_objects
from heat.objects import snapshot as snapshot_object
from heat.objects import stack as stack_object
from heat.rpc import api as rpc_api
from heat.rpc import worker_api as rpc_worker_api
@context.request_context
def stack_snapshot(self, cnxt, stack_identity, name):

    def _stack_snapshot(stack, snapshot):

        def save_snapshot(stack, action, status, reason):
            """Function that saves snapshot before snapshot complete."""
            data = stack.prepare_abandon()
            data['status'] = status
            snapshot_object.Snapshot.update(cnxt, snapshot.id, {'data': data, 'status': status, 'status_reason': reason})
        LOG.debug('Snapshotting stack %s', stack.name)
        stack.snapshot(save_snapshot_func=save_snapshot)
    s = self._get_stack(cnxt, stack_identity)
    stack = parser.Stack.load(cnxt, stack=s)
    if stack.status == stack.IN_PROGRESS:
        LOG.info('%(stack)s is in state %(action)s_IN_PROGRESS, snapshot is not permitted.', {'stack': str(stack), 'action': stack.action})
        raise exception.ActionInProgress(stack_name=stack.name, action=stack.action)
    if not cnxt.is_admin:
        stack_limit = cfg.CONF.max_snapshots_per_stack
        count_all = snapshot_object.Snapshot.count_all_by_stack(cnxt, stack.id)
        if stack_limit >= 0 and count_all >= stack_limit:
            message = _('You have reached the maximum snapshots per stack, %d. Please delete some snapshots.') % stack_limit
            raise exception.RequestLimitExceeded(message=message)
    lock = stack_lock.StackLock(cnxt, stack.id, self.engine_id)
    with lock.thread_lock():
        snapshot = snapshot_object.Snapshot.create(cnxt, {'tenant': cnxt.tenant_id, 'name': name, 'stack_id': stack.id, 'status': 'IN_PROGRESS'})
        self.thread_group_mgr.start_with_acquired_lock(stack, lock, _stack_snapshot, stack, snapshot)
        return api.format_snapshot(snapshot)