import copy
import functools
import getpass
import logging
import os
import time
import warnings
from cliff import columns as cliff_columns
from oslo_utils import importutils
from osc_lib import exceptions
from osc_lib.i18n import _
def format_list_of_dicts(data):
    """Return a formatted string of key value pairs for each dict

    :param data: a list of dicts
    :rtype: a string formatted to key='value' with dicts separated by new line
    """
    if data is None:
        return None
    return '\n'.join((format_dict(i) for i in data))