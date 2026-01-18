from heat.common import exception
from heat.common.i18n import _
from heat.engine import constraints as constr
from heat.engine import parameters
def set_stack_id(self, stack_identifier):
    """Set the StackId pseudo parameter value."""
    if stack_identifier is not None:
        self.params[self.PARAM_STACK_ID].schema.set_default(stack_identifier.stack_id)
        return True
    return False