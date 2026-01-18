import collections
import fnmatch
import glob
import itertools
import os.path
import re
import weakref
from oslo_config import cfg
from oslo_log import log
from heat.common import environment_format as env_fmt
from heat.common import exception
from heat.common.i18n import _
from heat.common import policy
from heat.engine import support
def is_hook_definition(key, value):
    is_valid_hook = False
    if key == 'hooks':
        if isinstance(value, str):
            is_valid_hook = valid_hook_type(value)
        elif isinstance(value, collections.abc.Sequence):
            is_valid_hook = all((valid_hook_type(hook) for hook in value))
        if not is_valid_hook:
            msg = _('Invalid hook type "%(value)s" for resource breakpoint, acceptable hook types are: %(types)s') % {'value': value, 'types': HOOK_TYPES}
            raise exception.InvalidBreakPointHook(message=msg)
    return is_valid_hook