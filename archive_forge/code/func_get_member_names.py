from heat.common import exception
from heat.common.i18n import _
from heat.engine import status
from heat.engine import template
from heat.rpc import api as rpc_api
def get_member_names(group):
    """Get a list of resource names of the resources in the specified group.

    Failed resources will be ignored.
    """
    inspector = GroupInspector.from_parent_resource(group)
    return list(inspector.member_names(include_failed=False))