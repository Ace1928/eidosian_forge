from typing import Dict, List, Tuple
from . import errors, registry
from .lazy_import import lazy_import
from breezy import (
from breezy.i18n import gettext
def add_hook(self, name, doc, introduced, deprecated=None):
    """Add a hook point to this dictionary.

        :param name: The name of the hook, for clients to use when registering.
        :param doc: The docs for the hook.
        :param introduced: When the hook was introduced (e.g. (0, 15)).
        :param deprecated: When the hook was deprecated, None for
            not-deprecated.
        """
    if name in self:
        raise errors.DuplicateKey(name)
    if self._module:
        callbacks = _lazy_hooks.setdefault((self._module, self._member_name, name), [])
    else:
        callbacks = None
    hookpoint = HookPoint(name=name, doc=doc, introduced=introduced, deprecated=deprecated, callbacks=callbacks)
    self[name] = hookpoint