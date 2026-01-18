import collections
import copy
import inspect
import logging
import pkgutil
import sys
import types
def find_patches(modules, recursive=True):
    """Find all the patches created through decorators.

    Parameters
    ----------
    modules : list of module
        Modules and/or packages to search the patches in.
    recursive : bool
        ``True`` to search recursively in subpackages.

    Returns
    -------
    list of gorilla.Patch
        Patches found.

    Raises
    ------
    TypeError
        The input is not a valid package or module.

    See Also
    --------
    :func:`patch`, :func:`patches`.
    """
    out = []
    modules = (module for package in modules for module in _module_iterator(package, recursive=recursive))
    for module in modules:
        members = _get_members(module, filter=None)
        for _, value in members:
            base = _get_base(value)
            decorator_data = get_decorator_data(base)
            if decorator_data is None:
                continue
            out.extend(decorator_data.patches)
    return out