import functools
import inspect
import wrapt
from debtcollector import _utils
def moved_class(new_class, old_class_name, old_module_name, message=None, version=None, removal_version=None, stacklevel=3, category=None):
    """Deprecates a class that was moved to another location.

    This creates a 'new-old' type that can be used for a
    deprecation period that can be inherited from. This will emit warnings
    when the old locations class is initialized, telling where the new and
    improved location for the old class now is.
    """
    if not inspect.isclass(new_class):
        _qual, type_name = _utils.get_qualified_name(type(new_class))
        raise TypeError("Unexpected class type '%s' (expected class type only)" % type_name)
    old_name = '.'.join((old_module_name, old_class_name))
    new_name = _utils.get_class_name(new_class)
    prefix = _CLASS_MOVED_PREFIX_TPL % (old_name, new_name)
    out_message = _utils.generate_message(prefix, message=message, version=version, removal_version=removal_version)

    def decorator(f):

        @functools.wraps(f, assigned=_utils.get_assigned(f))
        def wrapper(self, *args, **kwargs):
            _utils.deprecation(out_message, stacklevel=stacklevel, category=category)
            return f(self, *args, **kwargs)
        return wrapper
    old_class = type(old_class_name, (new_class,), {})
    old_class.__module__ = old_module_name
    old_class.__init__ = decorator(old_class.__init__)
    return old_class