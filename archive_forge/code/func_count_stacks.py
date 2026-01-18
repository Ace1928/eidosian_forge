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
def count_stacks(self, cnxt, filters=None, tenant_safe=True, show_deleted=False, show_nested=False, show_hidden=False, tags=None, tags_any=None, not_tags=None, not_tags_any=None):
    """Return the number of stacks that match the given filters.

        :param cnxt: RPC context.
        :param filters: a dict of ATTR:VALUE to match against stacks
        :param tenant_safe: DEPRECATED, if true, scope the request by
            the current tenant
        :param show_deleted: if true, count will include the deleted stacks
        :param show_nested: if true, count will include nested stacks
        :param show_hidden: if true, count will include hidden stacks
        :param tags: count stacks containing these tags. If multiple tags
            are passed, they will be combined using the boolean AND expression
        :param tags_any: count stacks containing these tags. If multiple tags
            are passed, they will be combined using the boolean OR expression
        :param not_tags: count stacks not containing these tags. If multiple
            tags are passed, they will be combined using the boolean AND
            expression
        :param not_tags_any: count stacks not containing these tags. If
            multiple tags are passed, they will be combined using the boolean
            OR expression
        :returns: an integer representing the number of matched stacks
        """
    if not tenant_safe:
        cnxt = context.get_admin_context()
    return stack_object.Stack.count_all(cnxt, filters=filters, show_deleted=show_deleted, show_nested=show_nested, show_hidden=show_hidden, tags=tags, tags_any=tags_any, not_tags=not_tags, not_tags_any=not_tags_any)