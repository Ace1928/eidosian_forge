from __future__ import annotations
import importlib
import sys
from typing import TYPE_CHECKING
import warnings
from pandas.util._exceptions import find_stack_level
from pandas.util.version import Version
def import_optional_dependency(name: str, extra: str='', errors: str='raise', min_version: str | None=None):
    """
    Import an optional dependency.

    By default, if a dependency is missing an ImportError with a nice
    message will be raised. If a dependency is present, but too old,
    we raise.

    Parameters
    ----------
    name : str
        The module name.
    extra : str
        Additional text to include in the ImportError message.
    errors : str {'raise', 'warn', 'ignore'}
        What to do when a dependency is not found or its version is too old.

        * raise : Raise an ImportError
        * warn : Only applicable when a module's version is to old.
          Warns that the version is too old and returns None
        * ignore: If the module is not installed, return None, otherwise,
          return the module, even if the version is too old.
          It's expected that users validate the version locally when
          using ``errors="ignore"`` (see. ``io/html.py``)
    min_version : str, default None
        Specify a minimum version that is different from the global pandas
        minimum version required.
    Returns
    -------
    maybe_module : Optional[ModuleType]
        The imported module, when found and the version is correct.
        None is returned when the package is not found and `errors`
        is False, or when the package's version is too old and `errors`
        is ``'warn'`` or ``'ignore'``.
    """
    assert errors in {'warn', 'raise', 'ignore'}
    package_name = INSTALL_MAPPING.get(name)
    install_name = package_name if package_name is not None else name
    msg = f"Missing optional dependency '{install_name}'. {extra} Use pip or conda to install {install_name}."
    try:
        module = importlib.import_module(name)
    except ImportError:
        if errors == 'raise':
            raise ImportError(msg)
        return None
    parent = name.split('.')[0]
    if parent != name:
        install_name = parent
        module_to_get = sys.modules[install_name]
    else:
        module_to_get = module
    minimum_version = min_version if min_version is not None else VERSIONS.get(parent)
    if minimum_version:
        version = get_version(module_to_get)
        if version and Version(version) < Version(minimum_version):
            msg = f"Pandas requires version '{minimum_version}' or newer of '{parent}' (version '{version}' currently installed)."
            if errors == 'warn':
                warnings.warn(msg, UserWarning, stacklevel=find_stack_level())
                return None
            elif errors == 'raise':
                raise ImportError(msg)
            else:
                return None
    return module