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
def get_default_dict(self):
    dd = self._default_by_name.copy()
    for field_name, default in self._default_callable_list:
        dd[field_name] = default()
    return dd