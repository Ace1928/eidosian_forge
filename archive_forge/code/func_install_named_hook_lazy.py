from typing import Dict, List, Tuple
from . import errors, registry
from .lazy_import import lazy_import
from breezy import (
from breezy.i18n import gettext
def install_named_hook_lazy(self, hook_name, callable_module, callable_member, name):
    """Install a_callable in to the hook hook_name lazily, and label it.

        :param hook_name: A hook name. See the __init__ method for the complete
            list of hooks.
        :param callable_module: Name of the module in which the callable is
            present.
        :param callable_member: Member name of the callable.
        :param name: A name to associate the callable with, to show users what
            is running.
        """
    try:
        hook = self[hook_name]
    except KeyError:
        raise UnknownHook(self.__class__.__name__, hook_name)
    try:
        hook_lazy = getattr(hook, 'hook_lazy')
    except AttributeError:
        raise errors.UnsupportedOperation(self.install_named_hook_lazy, self)
    else:
        hook_lazy(callable_module, callable_member, name)
    if name is not None:
        self.name_hook_lazy(callable_module, callable_member, name)