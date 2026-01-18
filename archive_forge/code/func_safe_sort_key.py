from collections import abc
import decimal
import random
import weakref
from oslo_log import log as logging
from oslo_utils import timeutils
from oslo_utils import uuidutils
from neutron_lib._i18n import _
def safe_sort_key(value):
    """Return value hash or build one for dictionaries.

    :param value: The value to build a hash for.
    :returns: The value sorted.
    """
    if isinstance(value, abc.Mapping):
        return sorted(value.items())
    return value