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
def get_rsrc_restricted_actions(self, resource_name):
    """Returns a set of restricted actions.

        For a given resource we get the set of restricted actions.

        Actions are set in this format via `resources`::

            {
                "restricted_actions": [update, replace]
            }

        A restricted_actions value is either `update`, `replace` or a list
        of those values. Resources support wildcard matching. The asterisk
        sign matches everything.
        """
    ress = self._registry['resources']
    restricted_actions = set()
    for name_pattern, resource in ress.items():
        if fnmatch.fnmatchcase(resource_name, name_pattern):
            if 'restricted_actions' in resource:
                actions = resource['restricted_actions']
                if isinstance(actions, str):
                    restricted_actions.add(actions)
                elif isinstance(actions, collections.abc.Sequence):
                    restricted_actions |= set(actions)
    return restricted_actions