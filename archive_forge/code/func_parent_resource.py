import itertools
from heat.common import exception
from heat.engine import attributes
from heat.engine import status
@property
def parent_resource(self):
    """Return a proxy for the parent resource.

        Returns None if the stack is not a provider stack for a
        TemplateResource.
        """
    return self._parent_info