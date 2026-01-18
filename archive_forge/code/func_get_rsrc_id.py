from heat.common import exception
from heat.common.i18n import _
from heat.engine import status
from heat.engine import template
from heat.rpc import api as rpc_api
def get_rsrc_id(stack, key, use_indices, resource_name):
    resource = get_resource(stack, resource_name, use_indices)
    if resource:
        return resource.FnGetRefId()