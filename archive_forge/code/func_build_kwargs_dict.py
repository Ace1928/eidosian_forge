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
def build_kwargs_dict(arg_name, value):
    """Return a dictionary containing `arg_name` if `value` is set."""
    kwargs = {}
    if value:
        kwargs[arg_name] = value
    return kwargs