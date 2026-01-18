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
def backward_compat_col_lister(column_headers, columns, column_map):
    """Convert the column headers to keep column backward compatibility.

    Replace the new column name of column headers by old name, so that
    the column headers can continue to support to show the old column name by
    --column/-c option with old name, like: volume list -c 'Display Name'

    :param column_headers: The column headers to be output in list command.
    :param columns: The columns to be output.
    :param column_map: The key of map is old column name, the value is new
            column name, like: {'old_col': 'new_col'}
    """
    if not columns:
        return column_headers
    column_headers = list(column_headers)
    for old_col, new_col in column_map.items():
        if old_col in columns:
            LOG.warning(_('The column "%(old_column)s" was deprecated, please use "%(new_column)s" replace.') % {'old_column': old_col, 'new_column': new_col})
            if new_col in column_headers:
                column_headers[column_headers.index(new_col)] = old_col
    return column_headers