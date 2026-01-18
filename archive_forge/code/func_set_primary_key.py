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
def set_primary_key(self, name, field):
    self.composite_key = isinstance(field, CompositeKey)
    self.add_field(name, field)
    self.primary_key = field
    self.auto_increment = field.auto_increment or bool(field.sequence)