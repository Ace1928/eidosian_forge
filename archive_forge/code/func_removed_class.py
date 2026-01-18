import functools
import inspect
import wrapt
from debtcollector import _utils
def removed_class(cls_name, replacement=None, message=None, version=None, removal_version=None, stacklevel=3, category=None):
    """Decorates a class to denote that it will be removed at some point."""

    def _wrap_it(old_init, out_message):

        @functools.wraps(old_init, assigned=_utils.get_assigned(old_init))
        def new_init(self, *args, **kwargs):
            _utils.deprecation(out_message, stacklevel=stacklevel, category=category)
            return old_init(self, *args, **kwargs)
        return new_init

    def _check_it(cls):
        if not inspect.isclass(cls):
            _qual, type_name = _utils.get_qualified_name(type(cls))
            raise TypeError("Unexpected class type '%s' (expected class type only)" % type_name)

    def _cls_decorator(cls):
        _check_it(cls)
        out_message = _utils.generate_message("Using class '%s' (either directly or via inheritance) is deprecated" % cls_name, postfix=None, message=message, version=version, removal_version=removal_version)
        cls.__init__ = _wrap_it(cls.__init__, out_message)
        return cls
    return _cls_decorator