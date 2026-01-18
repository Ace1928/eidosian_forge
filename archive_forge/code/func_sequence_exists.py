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
def sequence_exists(self, sequence):
    res = self.execute_sql("\n            SELECT COUNT(*) FROM pg_class, pg_namespace\n            WHERE relkind='S'\n                AND pg_class.relnamespace = pg_namespace.oid\n                AND relname=%s", (sequence,))
    return bool(res.fetchone()[0])