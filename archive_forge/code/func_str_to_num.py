import collections
import json
import numbers
import re
from oslo_cache import core
from oslo_config import cfg
from oslo_log import log
from oslo_utils import reflection
from oslo_utils import strutils
from heat.common import cache
from heat.common import exception
from heat.common.i18n import _
from heat.engine import resources
@staticmethod
def str_to_num(value):
    """Convert a string representation of a number into a numeric type."""
    if isinstance(value, numbers.Number):
        return value
    try:
        return int(value)
    except ValueError:
        return float(value)