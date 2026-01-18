import collections
import operator
import re
import warnings
import abc
from debtcollector import removals
import netaddr
import rfc3986
def format_defaults(self, default, sample_default=None):
    """Return a list of formatted default values.

        """
    if sample_default is not None:
        default_list = self._formatter(sample_default)
    elif not default:
        default_list = self.NONE_DEFAULT
    else:
        default_list = self._formatter(default)
    return default_list