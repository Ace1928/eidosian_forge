import functools
import inspect
import wrapt
from debtcollector import _utils
def moved_property(new_attribute_name, message=None, version=None, removal_version=None, stacklevel=3, category=None):
    """Decorates an *instance* property that was moved to another location."""
    return _moved_decorator('Property', new_attribute_name, message=message, version=version, removal_version=removal_version, stacklevel=stacklevel, category=category)