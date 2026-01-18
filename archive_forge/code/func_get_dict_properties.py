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
def get_dict_properties(item, fields, mixed_case_fields=None, formatters=None):
    """Return a tuple containing the item properties.

    :param item: a single dict resource
    :param fields: tuple of strings with the desired field names
    :param mixed_case_fields: tuple of field names to preserve case
    :param formatters: dictionary mapping field names to callables
       to format the values
    """
    if mixed_case_fields is None:
        mixed_case_fields = []
    if formatters is None:
        formatters = {}
    row = []
    for field in fields:
        if field in mixed_case_fields:
            field_name = field.replace(' ', '_')
        else:
            field_name = field.lower().replace(' ', '_')
        data = item[field_name] if field_name in item else ''
        if field in formatters:
            formatter = formatters[field]
            if isinstance(formatter, type) and issubclass(formatter, cliff_columns.FormattableColumn):
                data = formatter(data)
            elif isinstance(formatter, functools.partial) and isinstance(formatter.func, type) and issubclass(formatter.func, cliff_columns.FormattableColumn):
                data = formatter(data)
            elif callable(formatter):
                warnings.warn('The usage of formatter functions is now discouraged. Consider using cliff.columns.FormattableColumn instead. See reviews linked with bug 1687955 for more detail.', category=DeprecationWarning)
                if data is not None:
                    data = formatter(data)
            else:
                msg = 'Invalid formatter provided.'
                raise exceptions.CommandError(msg)
        row.append(data)
    return tuple(row)