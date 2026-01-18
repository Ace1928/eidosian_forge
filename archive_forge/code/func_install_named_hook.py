from typing import Dict, List, Tuple
from . import errors, registry
from .lazy_import import lazy_import
from breezy import (
from breezy.i18n import gettext
def install_named_hook(self, hook_name, a_callable, name):
    """Install a_callable in to the hook hook_name, and label it name.

        :param hook_name: A hook name. See the __init__ method for the complete
            list of hooks.
        :param a_callable: The callable to be invoked when the hook triggers.
            The exact signature will depend on the hook - see the __init__
            method for details on each hook.
        :param name: A name to associate a_callable with, to show users what is
            running.
        """
    try:
        hook = self[hook_name]
    except KeyError:
        raise UnknownHook(self.__class__.__name__, hook_name)
    try:
        hook.append(a_callable)
    except AttributeError:
        hook.hook(a_callable, name)
    if name is not None:
        self.name_hook(a_callable, name)