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
def is_valid_restricted_action(key, value):
    valid_action = False
    if key == 'restricted_actions':
        if isinstance(value, str):
            valid_action = valid_restricted_actions(value)
        elif isinstance(value, collections.abc.Sequence):
            valid_action = all((valid_restricted_actions(action) for action in value))
        if not valid_action:
            msg = _('Invalid restricted_action type "%(value)s" for resource, acceptable restricted_action types are: %(types)s') % {'value': value, 'types': RESTRICTED_ACTIONS}
            raise exception.InvalidRestrictedAction(message=msg)
    return valid_action