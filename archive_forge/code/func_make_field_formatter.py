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
def make_field_formatter(attr, filters=None):
    """Given an object attribute.

    Return a formatted field name and a formatter suitable for passing to
    print_list.
    Optionally pass a dict mapping attribute names to a function. The function
    will be passed the value of the attribute and should return the string to
    display.
    """
    filter_ = None
    if filters:
        filter_ = filters.get(attr)

    def get_field(obj):
        field = getattr(obj, attr, '')
        if field and filter_:
            field = filter_(field)
        return field
    name = _format_field_name(attr)
    formatter = get_field
    return (name, formatter)