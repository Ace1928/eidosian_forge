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
def format_dict_of_list(data, separator='; '):
    """Return a formatted string of key value pair

    :param data: a dict, key is string, value is a list of string, for example:
                 {u'public': [u'2001:db8::8', u'172.24.4.6']}
    :param separator: the separator to use between key/value pair
                      (default: '; ')
    :return: a string formatted to {'key1'=['value1', 'value2']} with separated
             by separator
    """
    if data is None:
        return None
    output = []
    for key in sorted(data):
        value = data[key]
        if value is None:
            continue
        value_str = format_list(value)
        group = '%s=%s' % (key, value_str)
        output.append(group)
    return separator.join(output)