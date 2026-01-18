import itertools
from heat.common import exception
from heat.engine import attributes
from heat.engine import status
def enabled_rsrc_names(self):
    """Return the set of names of all enabled resources in the template."""
    if self._resource_defns is None:
        self._load_rsrc_defns()
    return set(self._resource_defns)