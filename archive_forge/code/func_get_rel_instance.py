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
def get_rel_instance(self, instance):
    value = instance.__data__.get(self.name)
    if value is not None or self.name in instance.__rel__:
        if self.name not in instance.__rel__ and self.field.lazy_load:
            obj = self.rel_model.get(self.field.rel_field == value)
            instance.__rel__[self.name] = obj
        return instance.__rel__.get(self.name, value)
    elif not self.field.null and self.field.lazy_load:
        raise self.rel_model.DoesNotExist
    return value