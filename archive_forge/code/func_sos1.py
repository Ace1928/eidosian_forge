from pyomo.core.expr.numvalue import is_numeric_data
from pyomo.core.kernel.base import ICategorizedObject, _abstract_readwrite_property
from pyomo.core.kernel.container_utils import define_simple_containers
def sos1(variables, weights=None):
    """A Special Ordered Set of type 1.

    This is an alias for sos(..., level=1)"""
    return sos(variables, weights=weights, level=1)