from collections import abc
import decimal
import random
import weakref
from oslo_log import log as logging
from oslo_utils import timeutils
from oslo_utils import uuidutils
from neutron_lib._i18n import _
def make_weak_ref(f):
    """Make a weak reference to a function accounting for bound methods.

    :param f: The callable to make a weak ref for.
    :returns: A weak ref to f.
    """
    return weak_method(f) if hasattr(f, '__self__') else weakref.ref(f)