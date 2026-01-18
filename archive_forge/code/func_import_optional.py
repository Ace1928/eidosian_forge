from __future__ import annotations
import logging # isort:skip
from importlib import import_module
from types import ModuleType
from typing import Any
def import_optional(mod_name: str) -> ModuleType | None:
    """ Attempt to import an optional dependency.

    Silently returns None if the requested module is not available.

    Args:
        mod_name (str) : name of the optional module to try to import

    Returns:
        imported module or None, if import fails

    """
    try:
        return import_module(mod_name)
    except ImportError:
        pass
    except Exception:
        msg = f'Failed to import optional module `{mod_name}`'
        log.exception(msg)
    return None