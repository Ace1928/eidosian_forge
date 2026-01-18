import collections.abc
import contextlib
import datetime
import functools
import inspect
import io
import os
import re
import socket
import sys
import threading
import types
import enum
from oslo_serialization import jsonutils
from oslo_serialization import msgpackutils
from oslo_utils import encodeutils
from oslo_utils import importutils
from oslo_utils import netutils
from oslo_utils import reflection
from taskflow.types import failure
def merge_uri(uri, conf):
    """Merges a parsed uri into the given configuration dictionary.

    Merges the username, password, hostname, port, and query parameters of
    a URI into the given configuration dictionary (it does **not** overwrite
    existing configuration keys if they already exist) and returns the merged
    configuration.

    NOTE(harlowja): does not merge the path, scheme or fragment.
    """
    uri_port = uri.port
    specials = [('username', uri.username, lambda v: bool(v)), ('password', uri.password, lambda v: bool(v)), ('port', uri_port, lambda v: v is not None)]
    hostname = uri.hostname
    if hostname:
        if uri_port is not None:
            hostname += ':%s' % uri_port
        specials.append(('hostname', hostname, lambda v: bool(v)))
    for k, v, is_not_empty_value_func in specials:
        if is_not_empty_value_func(v):
            conf.setdefault(k, v)
    for k, v in uri.params().items():
        conf.setdefault(k, v)
    return conf