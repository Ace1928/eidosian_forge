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
def group_by_extend(self, *values):
    """@Node.copy used from group_by() call"""
    group_by = tuple(self._group_by or ()) + values
    return self.group_by(*group_by)