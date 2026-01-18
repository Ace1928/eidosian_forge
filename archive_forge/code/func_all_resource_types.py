import itertools
from heat.common import exception
from heat.engine import attributes
from heat.engine import status
def all_resource_types(self):
    """Return the set of types of all resources in the template."""
    if self._resource_defns is None:
        self._load_rsrc_defns()
    return set((self._resource_defns[res].resource_type for res in self._resource_defns))