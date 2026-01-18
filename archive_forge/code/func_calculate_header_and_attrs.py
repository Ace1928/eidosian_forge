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
def calculate_header_and_attrs(column_headers, attrs, parsed_args):
    """Calculate headers and attribute names based on parsed_args.column.

    When --column (-c) option is specified, this function calculates
    column headers and expected API attribute names according to
    the OSC header/column definitions.

    This function also adjusts the content of parsed_args.columns
    if API attribute names are used in parsed_args.columns.
    This allows users to specify API attribute names in -c option.

    :param column_headers: A tuple/list of column headers to display
    :param attrs: a tuple/list of API attribute names. The order of
        corresponding column header and API attribute name must match.
    :param parsed_args: Parsed argument object returned by argparse parse_args
    :returns: A tuple of calculated headers and API attribute names.
    """
    if parsed_args.columns:
        header_attr_map = dict(zip(column_headers, attrs))
        expected_attrs = [header_attr_map.get(c, c) for c in parsed_args.columns]
        attr_header_map = dict(zip(attrs, column_headers))
        expected_headers = [attr_header_map.get(c, c) for c in parsed_args.columns]
        parsed_args.columns = expected_headers
        return (expected_headers, expected_attrs)
    else:
        return (column_headers, attrs)