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
def matches_hook(self, resource_name, hook):
    """Return whether a resource have a hook set in the environment.

        For a given resource and a hook type, we check to see if the passed
        group of resources has the right hook associated with the name.

        Hooks are set in this format via `resources`::

            {
                "res_name": {
                    "hooks": [pre-create, pre-update]
                },
                "*_suffix": {
                    "hooks": pre-create
                },
                "prefix_*": {
                    "hooks": pre-update
                }
            }

        A hook value is either `pre-create`, `pre-update` or a list of those
        values. Resources support wildcard matching. The asterisk sign matches
        everything.
        """
    ress = self._registry['resources']
    for name_pattern, resource in ress.items():
        if fnmatch.fnmatchcase(resource_name, name_pattern):
            if 'hooks' in resource:
                hooks = resource['hooks']
                if isinstance(hooks, str):
                    if hook == hooks:
                        return True
                elif isinstance(hooks, collections.abc.Sequence):
                    if hook in hooks:
                        return True
    return False