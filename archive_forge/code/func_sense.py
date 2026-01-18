from pyomo.core.expr.numvalue import as_numeric
from pyomo.core.kernel.base import _abstract_readwrite_property
from pyomo.core.kernel.container_utils import define_simple_containers
from pyomo.core.kernel.expression import IExpression
@sense.setter
def sense(self, sense):
    """Set the sense (direction) of this objective."""
    if sense == minimize or sense == maximize:
        self._sense = sense
    else:
        raise ValueError("Objective sense must be set to one of: [minimize (%s), maximize (%s)]. Invalid value: %s'" % (minimize, maximize, sense))