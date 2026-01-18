import base64
import logging
import os
import textwrap
import uuid
from oslo_serialization import jsonutils
from oslo_utils import encodeutils
import prettytable
from urllib import error
from urllib import parse
from urllib import request
import yaml
from heatclient._i18n import _
from heatclient import exc
def print_update_list(lst, fields, formatters=None):
    """Print the stack-update --dry-run output as a table.

    This function is necessary to print the stack-update --dry-run
    output, which contains additional information about the update.
    """
    formatters = formatters or {}
    pt = prettytable.PrettyTable(fields, caching=False, print_empty=False)
    pt.align = 'l'
    for change in lst:
        row = []
        for field in fields:
            if field in formatters:
                row.append(formatters[field](change.get(field, None)))
            else:
                row.append(change.get(field, None))
        pt.add_row(row)
    print(encodeutils.safe_encode(pt.get_string()).decode())