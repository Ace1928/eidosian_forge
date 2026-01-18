from typing import Dict, List, Tuple
from . import errors, registry
from .lazy_import import lazy_import
from breezy import (
from breezy.i18n import gettext
def known_hooks_key_to_object(key):
    """Convert a known_hooks key to a object.

    :param key: A tuple (module_name, member_name) as found in the keys of
        the known_hooks registry.
    :return: The object this specifies.
    """
    return pyutils.get_named_object(*key)