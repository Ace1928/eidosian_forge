from numbers import Number, Integral
from .. import _api_internal
def convert_to_node(value):
    """Convert a python value to corresponding node type.

    Parameters
    ----------
    value : str
        The value to be inspected.

    Returns
    -------
    node : Node
        The corresponding node value.
    """
    if isinstance(value, Integral):
        return _api_internal._Integer(value)
    elif isinstance(value, (list, tuple)):
        value = [convert_to_node(x) for x in value]
        return _api_internal._ADT(*value)
    raise ValueError("don't know how to convert type %s to node" % type(value))