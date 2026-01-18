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
def execute_returning(self, database):
    if self._cursor_wrapper is None:
        cursor = database.execute(self)
        self._cursor_wrapper = self._get_cursor_wrapper(cursor)
    return self._cursor_wrapper