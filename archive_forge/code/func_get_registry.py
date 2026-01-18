import json
import warnings
from .base import string_types
def get_registry(base_class):
    """Get a copy of the registry.

    Parameters
    ----------
    base_class : type
        base class for classes that will be registered.

    Returns
    -------
    a registrator
    """
    if base_class not in _REGISTRY:
        _REGISTRY[base_class] = {}
    return _REGISTRY[base_class].copy()