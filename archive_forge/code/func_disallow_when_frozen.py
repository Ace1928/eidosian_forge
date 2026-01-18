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
def disallow_when_frozen(excp_cls):
    """Frozen checking/raising method decorator."""

    def decorator(f):

        @functools.wraps(f)
        def wrapper(self, *args, **kwargs):
            if self.frozen:
                raise excp_cls()
            else:
                return f(self, *args, **kwargs)
        return wrapper
    return decorator