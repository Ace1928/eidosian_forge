import abc
import copy
import functools
import urllib
import warnings
from keystoneauth1 import exceptions as ksa_exceptions
from keystoneauth1 import plugin
from oslo_utils import strutils
from keystoneclient import exceptions as ksc_exceptions
from keystoneclient.i18n import _
def filter_none(**kwargs):
    """Remove any entries from a dictionary where the value is None."""
    return dict(((k, v) for k, v in kwargs.items() if v is not None))