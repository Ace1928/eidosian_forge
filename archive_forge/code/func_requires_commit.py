import abc
import collections
import inspect
import itertools
import operator
import typing as ty
import urllib.parse
import warnings
import jsonpatch
from keystoneauth1 import adapter
from keystoneauth1 import discover
from requests import structures
from openstack import _log
from openstack import exceptions
from openstack import format
from openstack import utils
from openstack import warnings as os_warnings
@property
def requires_commit(self):
    """Whether the next commit() call will do anything."""
    return self._body.dirty or self._header.dirty or self.allow_empty_commit