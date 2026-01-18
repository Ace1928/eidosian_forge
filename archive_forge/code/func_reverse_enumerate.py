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
def reverse_enumerate(items):
    """Like reversed(enumerate(items)) but with less copying/cloning..."""
    for i in countdown_iter(len(items)):
        yield (i - 1, items[i - 1])