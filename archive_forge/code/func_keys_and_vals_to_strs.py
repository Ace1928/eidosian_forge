import getpass
import inspect
import os
import sys
import textwrap
import decorator
from magnumclient.common.apiclient import exceptions
from oslo_utils import encodeutils
from oslo_utils import strutils
import prettytable
from magnumclient.i18n import _
def keys_and_vals_to_strs(dictionary):
    """Recursively convert a dictionary's keys and values to strings.

    :param dictionary: dictionary whose keys/vals are to be converted to strs
    """

    def to_str(k_or_v):
        if isinstance(k_or_v, dict):
            return keys_and_vals_to_strs(k_or_v)
        elif isinstance(k_or_v, str):
            return str(k_or_v)
        else:
            return k_or_v
    return dict(((to_str(k), to_str(v)) for k, v in dictionary.items()))