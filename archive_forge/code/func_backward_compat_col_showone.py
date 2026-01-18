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
def backward_compat_col_showone(show_object, columns, column_map):
    """Convert the output object to keep column backward compatibility.

    Replace the new column name of output object by old name, so that
    the object can continue to support to show the old column name by
    --column/-c option with old name, like: volume show -c 'display_name'

    :param show_object: The object to be output in create/show commands.
    :param columns: The columns to be output.
    :param column_map: The key of map is old column name, the value is new
        column name, like: {'old_col': 'new_col'}
    """
    if not columns:
        return show_object
    show_object = copy.deepcopy(show_object)
    for old_col, new_col in column_map.items():
        if old_col in columns:
            LOG.warning(_('The column "%(old_column)s" was deprecated, please use "%(new_column)s" replace.') % {'old_column': old_col, 'new_column': new_col})
            if new_col in show_object:
                show_object.update({old_col: show_object.pop(new_col)})
    return show_object