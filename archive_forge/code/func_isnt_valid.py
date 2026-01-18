from __future__ import annotations
import collections
import contextlib
import functools
import itertools
import os
import socket
import sys
import threading
from debugpy.common import json, log, util
from debugpy.common.util import hide_thread_from_debugger
def isnt_valid(self, *args, **kwargs):
    """Same as self.error(InvalidMessageError, ...)."""
    return self.error(InvalidMessageError, *args, **kwargs)