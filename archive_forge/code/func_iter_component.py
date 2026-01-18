from pyomo.core.base.reference import Reference
from pyomo.common.collections import ComponentMap
from pyomo.common.modeling import unique_component_name
def iter_component(obj):
    """
    Yield "child" objects from a component that is defined with either the `base` or `kernel` APIs.
    If the component is not indexed, it returns itself.

    Parameters
    ----------
    obj : ComponentType
        eg. `TupleContainer`, `ListContainer`, `DictContainer`, `IndexedComponent`, or `Component`

    Returns
    -------
    Iterator[ComponentType] : Iterator of the component data objects.
    """
    try:
        return iter(obj.values())
    except AttributeError:
        pass
    try:
        return iter(obj)
    except TypeError:
        return iter((obj,))