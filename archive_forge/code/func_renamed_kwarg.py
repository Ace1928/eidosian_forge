import wrapt
from debtcollector import _utils
def renamed_kwarg(old_name, new_name, message=None, version=None, removal_version=None, stacklevel=3, category=None, replace=False):
    """Decorates a kwarg accepting function to deprecate a renamed kwarg."""
    prefix = _KWARG_RENAMED_PREFIX_TPL % old_name
    postfix = _KWARG_RENAMED_POSTFIX_TPL % new_name
    out_message = _utils.generate_message(prefix, postfix=postfix, message=message, version=version, removal_version=removal_version)

    @wrapt.decorator
    def decorator(wrapped, instance, args, kwargs):
        if old_name in kwargs:
            _utils.deprecation(out_message, stacklevel=stacklevel, category=category)
            if replace:
                kwargs.setdefault(new_name, kwargs.pop(old_name))
        return wrapped(*args, **kwargs)
    return decorator