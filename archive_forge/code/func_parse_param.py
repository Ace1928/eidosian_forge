import collections
from oslo_log import log as logging
from oslo_serialization import jsonutils
from heat.common import environment_format as env_fmt
from heat.common import exception
from heat.common.i18n import _
def parse_param(p_val, p_schema):
    try:
        if p_schema.type == p_schema.MAP:
            if not isinstance(p_val, str):
                p_val = jsonutils.dumps(p_val)
            if p_val:
                return jsonutils.loads(p_val)
        elif not isinstance(p_val, collections.abc.Sequence):
            raise ValueError()
    except (ValueError, TypeError) as err:
        msg = _('Invalid parameter in environment %s.') % str(err)
        raise ValueError(msg)
    return p_val