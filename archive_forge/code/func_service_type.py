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
def service_type(stype):
    """Adds 'service_type' attribute to decorated function.

    Usage:

    .. code-block:: python

       @service_type('volume')
       def mymethod(f):
       ...
    """

    def inner(f):
        f.service_type = stype
        return f
    return inner