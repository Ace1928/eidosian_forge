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
def fields_to_index(self):
    indexes = []
    for f in self.sorted_fields:
        if f.primary_key:
            continue
        if f.index or f.unique:
            indexes.append(ModelIndex(self.model, (f,), unique=f.unique, using=f.index_type))
    for index_obj in self.indexes:
        if isinstance(index_obj, Node):
            indexes.append(index_obj)
        elif isinstance(index_obj, (list, tuple)):
            index_parts, unique = index_obj
            fields = []
            for part in index_parts:
                if isinstance(part, basestring):
                    fields.append(self.combined[part])
                elif isinstance(part, Node):
                    fields.append(part)
                else:
                    raise ValueError('Expected either a field name or a subclass of Node. Got: %s' % part)
            indexes.append(ModelIndex(self.model, fields, unique=unique))
    return indexes