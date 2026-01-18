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
def get_context_options(self):
    return {'field_types': self._field_types, 'operations': self._operations, 'param': self.param, 'quote': self.quote, 'compound_select_parentheses': self.compound_select_parentheses, 'conflict_statement': self.conflict_statement, 'conflict_update': self.conflict_update, 'for_update': self.for_update, 'index_schema_prefix': self.index_schema_prefix, 'index_using_precedes_table': self.index_using_precedes_table, 'limit_max': self.limit_max, 'nulls_ordering': self.nulls_ordering}