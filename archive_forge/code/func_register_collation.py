from bisect import bisect_left
from bisect import bisect_right
from contextlib import contextmanager
from copy import deepcopy
from functools import wraps
from inspect import isclass
import calendar
import collections
import datetime
import decimal
import hashlib
import itertools
import logging
import operator
import re
import socket
import struct
import sys
import threading
import time
import uuid
import warnings
def register_collation(self, fn, name=None):
    name = name or fn.__name__

    def _collation(*args):
        expressions = args + (SQL('collate %s' % name),)
        return NodeList(expressions)
    fn.collation = _collation
    self._collations[name] = fn
    if not self.is_closed():
        self._load_collations(self.connection())