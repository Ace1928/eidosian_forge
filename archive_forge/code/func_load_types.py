from ._dill import (
from .session import (
from . import detect, logger, session, source, temp
from .settings import settings
from . import objtypes as types
def load_types(pickleable=True, unpickleable=True):
    """load pickleable and/or unpickleable types to ``dill.types``

    ``dill.types`` is meant to mimic the ``types`` module, providing a
    registry of object types.  By default, the module is empty (for import
    speed purposes). Use the ``load_types`` function to load selected object
    types to the ``dill.types`` module.

    Args:
        pickleable (bool, default=True): if True, load pickleable types.
        unpickleable (bool, default=True): if True, load unpickleable types.

    Returns:
        None
    """
    from importlib import reload
    from . import _objects
    if pickleable:
        objects.update(_objects.succeeds)
    else:
        [objects.pop(obj, None) for obj in _objects.succeeds]
    if unpickleable:
        objects.update(_objects.failures)
    else:
        [objects.pop(obj, None) for obj in _objects.failures]
    objects.update(_objects.registered)
    del _objects
    [types.__dict__.pop(obj) for obj in list(types.__dict__.keys()) if obj.find('Type') != -1]
    reload(types)