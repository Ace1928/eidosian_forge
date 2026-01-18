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
def print_dict(dct, dict_property='Property', wrap=0):
    """Print a `dict` as a table of two columns.

    :param dct: `dict` to print
    :param dict_property: name of the first column
    :param wrap: wrapping for the second column
    """
    pt = prettytable.PrettyTable([dict_property, 'Value'])
    pt.align = 'l'
    for k, v in dct.items():
        if isinstance(v, dict):
            v = str(keys_and_vals_to_strs(v))
        if wrap > 0:
            v = textwrap.fill(str(v), wrap)
        if v and isinstance(v, str) and ('\\n' in v):
            lines = v.strip().split('\\n')
            col1 = k
            for line in lines:
                pt.add_row([col1, line])
                col1 = ''
        elif isinstance(v, list):
            val = str([str(i) for i in v])
            if val is None:
                val = '-'
            pt.add_row([k, val])
        else:
            if v is None:
                v = '-'
            pt.add_row([k, v])
    print(encodeutils.safe_encode(pt.get_string()).decode())