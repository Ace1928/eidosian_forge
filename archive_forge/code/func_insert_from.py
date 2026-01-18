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
@classmethod
def insert_from(cls, query, fields):
    columns = [getattr(cls, field) if isinstance(field, basestring) else field for field in fields]
    return ModelInsert(cls, insert=query, columns=columns)