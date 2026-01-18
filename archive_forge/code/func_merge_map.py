import collections
from oslo_log import log as logging
from oslo_serialization import jsonutils
from heat.common import environment_format as env_fmt
from heat.common import exception
from heat.common.i18n import _
def merge_map(old, new, deep_merge=False):
    """Merge nested dictionaries."""
    if not old:
        return new
    for k, v in new.items():
        if v is not None:
            if not deep_merge:
                old[k] = v
            elif isinstance(v, collections.abc.Mapping):
                old_v = old.get(k)
                old[k] = merge_map(old_v, v, deep_merge) if old_v else v
            elif isinstance(v, collections.abc.Sequence) and (not isinstance(v, str)):
                old_v = old.get(k)
                old[k] = merge_list(old_v, v) if old_v else v
            elif isinstance(v, str):
                old[k] = ''.join([old.get(k, ''), v])
            else:
                old[k] = v
    return old