import itertools
from heat.common import exception
from heat.engine import attributes
from heat.engine import status
def clone_with_new_template(self, new_template, stack_identifier, clear_resource_data=False):
    """Create a new StackDefinition with a different template."""
    res_data = {} if clear_resource_data else dict(self._resource_data)
    return type(self)(self._context, new_template, stack_identifier, res_data, self._parent_info)