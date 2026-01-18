from typing import Dict, List, Tuple
from . import errors, registry
from .lazy_import import lazy_import
from breezy import (
from breezy.i18n import gettext
def iter_parent_objects(self):
    """Yield (hook_key, (parent_object, attr)) tuples for every registered
        hook, where 'parent_object' is the object that holds the hook
        instance.

        This is useful for resetting/restoring all the hooks to a known state,
        as is done in breezy.tests.TestCase._clear_hooks.
        """
    for key in self.keys():
        yield (key, self.key_to_parent_and_attribute(key))