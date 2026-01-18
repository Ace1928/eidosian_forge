import collections
import copy
import inspect
import logging
import pkgutil
import sys
import types
def get_original_attribute(obj, name, bypass_descriptor_protocol=False):
    """Retrieve an overridden attribute that has been stored.

    Parameters
    ----------
    obj : object
        Object to search the attribute in.
    name : str
        Name of the attribute.
    bypass_descriptor_protocol: boolean
        bypassing descriptor protocol if true. When storing original method during patching or
        restoring original method during reverting patch, we need set bypass_descriptor_protocol
        to be True to ensure get the raw attribute object.

    Returns
    -------
    object
        The attribute found.

    Raises
    ------
    AttributeError
        The attribute couldn't be found.

    Note
    ----
    if setting store_hit=False, then after patch applied, this methods may return patched
    attribute instead of original attribute in specific cases.

    See Also
    --------
    :attr:`Settings.allow_hit`.
    """
    original_name = _ORIGINAL_NAME % (name,)
    curr_active_patch = _ACTIVE_PATCH % (name,)

    def _get_attr(obj_, name_):
        if bypass_descriptor_protocol:
            return object.__getattribute__(obj_, name_)
        else:
            return getattr(obj_, name_)
    no_original_stored_err = 'Original attribute %s was not stored when patching, set store_hit=True will address this.'
    if inspect.isclass(obj):
        for obj_ in inspect.getmro(obj):
            if original_name in obj_.__dict__:
                return _get_attr(obj_, original_name)
            elif name in obj_.__dict__:
                if curr_active_patch in obj_.__dict__:
                    patch = getattr(obj, curr_active_patch)
                    if patch.is_inplace_patch:
                        raise RuntimeError(no_original_stored_err % (f'{obj_.__name__}.{name}',))
                    else:
                        continue
                return _get_attr(obj_, name)
            else:
                continue
        raise AttributeError(f"'{type(obj)}' object has no attribute '{name}'")
    else:
        try:
            return _get_attr(obj, original_name)
        except AttributeError:
            if curr_active_patch in obj.__dict__:
                raise RuntimeError(no_original_stored_err % (f'{type(obj).__name__}.{name}',))
            return _get_attr(obj, name)