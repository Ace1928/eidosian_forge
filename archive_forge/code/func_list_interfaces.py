import inspect
import importlib
from ..interfaces.base import Interface
def list_interfaces(module):
    """Return a list with the names of the Interface subclasses inside
    the given module.
    """
    iface_names = []
    for k, v in sorted(list(module.__dict__.items())):
        if inspect.isclass(v) and issubclass(v, Interface):
            iface_names.append(k)
    return iface_names