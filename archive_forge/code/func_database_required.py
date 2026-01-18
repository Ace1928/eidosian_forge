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
def database_required(method):

    @wraps(method)
    def inner(self, database=None, *args, **kwargs):
        database = self._database if database is None else database
        if not database:
            raise InterfaceError('Query must be bound to a database in order to call "%s".' % method.__name__)
        return method(self, database, *args, **kwargs)
    return inner