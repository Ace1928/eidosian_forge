import itertools
from heat.common import exception
from heat.engine import attributes
from heat.engine import status
def resource_definition(self, resource_name):
    """Return the definition of the given resource."""
    if self._resource_defns is None:
        self._load_rsrc_defns()
    return self._resource_defns[resource_name]