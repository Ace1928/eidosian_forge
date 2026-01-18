from typing import Dict, List, Tuple
from . import errors, registry
from .lazy_import import lazy_import
from breezy import (
from breezy.i18n import gettext
def key_to_parent_and_attribute(self, key):
    """Convert a known_hooks key to a (parent_obj, attr) pair.

        :param key: A tuple (module_name, member_name) as found in the keys of
            the known_hooks registry.
        :return: The parent_object of the hook and the name of the attribute on
            that parent object where the hook is kept.
        """
    parent_mod, parent_member, attr = pyutils.calc_parent_name(*key)
    return (pyutils.get_named_object(parent_mod, parent_member), attr)