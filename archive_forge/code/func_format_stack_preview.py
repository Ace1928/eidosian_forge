import collections
from oslo_log import log as logging
from oslo_utils import timeutils
from heat.common.i18n import _
from heat.common import param_utils
from heat.common import template_format
from heat.common import timeutils as heat_timeutils
from heat.engine import constraints as constr
from heat.rpc import api as rpc_api
def format_stack_preview(stack):

    def format_resource(res):
        if isinstance(res, list):
            return list(map(format_resource, res))
        return format_stack_resource(res, with_props=True)
    fmt_stack = format_stack(stack, preview=True)
    fmt_resources = list(map(format_resource, stack.preview_resources()))
    fmt_stack['resources'] = fmt_resources
    return fmt_stack