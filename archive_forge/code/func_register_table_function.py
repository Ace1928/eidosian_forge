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
def register_table_function(self, klass, name=None):
    if name is not None:
        klass.name = name
    self._table_functions.append(klass)
    if not self.is_closed():
        klass.register(self.connection())