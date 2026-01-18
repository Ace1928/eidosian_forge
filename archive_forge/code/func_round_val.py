from collections import abc
import decimal
import random
import weakref
from oslo_log import log as logging
from oslo_utils import timeutils
from oslo_utils import uuidutils
from neutron_lib._i18n import _
def round_val(val):
    """Round the value.

    :param val: The value to round.
    :returns: The value rounded using the half round up scheme.
    """
    return int(decimal.Decimal(val).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP))