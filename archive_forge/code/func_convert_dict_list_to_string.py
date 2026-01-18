import getpass
import inspect
import os
import sys
import textwrap
from oslo_utils import encodeutils
from oslo_utils import strutils
import prettytable
from manilaclient.common._i18n import _
def convert_dict_list_to_string(data, ignored_keys=None):
    ignored_keys = ignored_keys or []
    if not isinstance(data, list):
        data = [data]
    data_string = ''
    for datum in data:
        if hasattr(datum, '_info'):
            datum_dict = datum._info
        else:
            datum_dict = datum
        for k, v in datum_dict.items():
            if k not in ignored_keys:
                data_string += '\n%(k)s = %(v)s' % {'k': k, 'v': v}
    return data_string