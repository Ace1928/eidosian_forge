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
def session_rollback(self):
    with self._lock:
        try:
            txn = self.pop_transaction()
        except IndexError:
            return False
        txn.rollback(begin=self.in_transaction())
        return True