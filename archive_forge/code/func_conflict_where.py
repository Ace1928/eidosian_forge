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
@Node.copy
def conflict_where(self, *expressions):
    if self._conflict_where is not None:
        expressions = (self._conflict_where,) + expressions
    self._conflict_where = reduce(operator.and_, expressions)