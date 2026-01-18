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
def suitability(ext_rsrc):
    score = 0
    if ext_rsrc.status == status.ResourceStatus.FAILED:
        score -= 30
    if ext_rsrc.action == status.ResourceStatus.DELETE:
        score -= 50
    if ext_rsrc.replaced_by:
        score -= 1
    if ext_rsrc.current_template_id == self.prev_raw_template_id:
        score += 5
    if ext_rsrc.current_template_id == self.t.id:
        score += 10
    last_changed_at = ext_rsrc.updated_at
    if last_changed_at is None:
        last_changed_at = ext_rsrc.created_at
    return (score, last_changed_at)