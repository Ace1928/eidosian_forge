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
@property
def safe_field_names(self):
    if self._safe_field_names is None:
        if self.model is None:
            return self.field_names
        self._safe_field_names = [self.model._meta.fields[f].safe_name for f in self.field_names]
    return self._safe_field_names