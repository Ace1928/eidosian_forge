import collections
from oslo_log import log as logging
from oslo_utils import timeutils
from heat.common.i18n import _
from heat.common import param_utils
from heat.common import template_format
from heat.common import timeutils as heat_timeutils
from heat.engine import constraints as constr
from heat.rpc import api as rpc_api
def format_software_config(sc, detail=True, include_project=False):
    if sc is None:
        return
    result = {rpc_api.SOFTWARE_CONFIG_ID: sc.id, rpc_api.SOFTWARE_CONFIG_NAME: sc.name, rpc_api.SOFTWARE_CONFIG_GROUP: sc.group, rpc_api.SOFTWARE_CONFIG_CREATION_TIME: heat_timeutils.isotime(sc.created_at)}
    if detail:
        result[rpc_api.SOFTWARE_CONFIG_CONFIG] = sc.config['config']
        result[rpc_api.SOFTWARE_CONFIG_INPUTS] = sc.config['inputs']
        result[rpc_api.SOFTWARE_CONFIG_OUTPUTS] = sc.config['outputs']
        result[rpc_api.SOFTWARE_CONFIG_OPTIONS] = sc.config['options']
    if include_project:
        result[rpc_api.SOFTWARE_CONFIG_PROJECT] = sc.tenant
    return result