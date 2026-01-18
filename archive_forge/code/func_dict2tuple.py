from collections import abc
import decimal
import random
import weakref
from oslo_log import log as logging
from oslo_utils import timeutils
from oslo_utils import uuidutils
from neutron_lib._i18n import _
def dict2tuple(d):
    """Build a tuple from a dict.

    :param d: The dict to coherence into a tuple.
    :returns: The dict d in tuple form.
    """
    items = list(d.items())
    items.sort()
    return tuple(items)