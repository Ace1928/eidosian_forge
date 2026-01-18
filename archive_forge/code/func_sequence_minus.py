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
def sequence_minus(seq1, seq2):
    """Calculate difference of two sequences.

    Result contains the elements from first sequence that are not
    present in second sequence, in original order. Works even
    if sequence elements are not hashable.
    """
    result = list(seq1)
    for item in seq2:
        try:
            result.remove(item)
        except ValueError:
            pass
    return result