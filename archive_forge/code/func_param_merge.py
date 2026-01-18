import collections
from oslo_log import log as logging
from oslo_serialization import jsonutils
from heat.common import environment_format as env_fmt
from heat.common import exception
from heat.common.i18n import _
def param_merge(p_key, p_value, p_schema, deep_merge=False):
    p_type = p_schema.type
    p_value = parse_param(p_value, p_schema)
    if p_type == p_schema.MAP:
        old[p_key] = merge_map(old.get(p_key, {}), p_value, deep_merge)
    elif p_type == p_schema.LIST:
        old[p_key] = merge_list(old.get(p_key), p_value)
    elif p_type == p_schema.STRING:
        old[p_key] = ''.join([old.get(p_key, ''), p_value])
    elif p_type == p_schema.NUMBER:
        old[p_key] = old.get(p_key, 0) + p_value
    else:
        raise exception.InvalidMergeStrategyForParam(strategy=MERGE, param=p_key)