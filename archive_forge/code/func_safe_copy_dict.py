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
def safe_copy_dict(obj):
    """Copy an existing dictionary or default to empty dict...

    This will return a empty dict if given object is falsey, otherwise it
    will create a dict of the given object (which if provided a dictionary
    object will make a shallow copy of that object).
    """
    if not obj:
        return {}
    return dict(obj)