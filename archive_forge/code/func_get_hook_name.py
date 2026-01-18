from typing import Dict, List, Tuple
from . import errors, registry
from .lazy_import import lazy_import
from breezy import (
from breezy.i18n import gettext
def get_hook_name(self, a_callable):
    """Get the name for a_callable for UI display.

        If no name has been registered, the string 'No hook name' is returned.
        We use a fixed string rather than repr or the callables module because
        the code names are rarely meaningful for end users and this is not
        intended for debugging.
        """
    name = self._callable_names.get(a_callable, None)
    if name is None and a_callable is not None:
        name = self._lazy_callable_names.get((a_callable.__module__, a_callable.__name__), None)
    if name is None:
        return 'No hook name'
    return name