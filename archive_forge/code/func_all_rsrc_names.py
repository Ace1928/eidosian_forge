import itertools
from heat.common import exception
from heat.engine import attributes
from heat.engine import status
def all_rsrc_names(self):
    """Return the set of names of all resources in the template.

        This includes resources that are disabled due to false conditionals.
        """
    if hasattr(self._template, 'RESOURCES'):
        return set(self._template.get(self._template.RESOURCES, self._resource_defns or []))
    else:
        return self.enabled_rsrc_names()