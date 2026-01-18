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
def populate_instance(self, instance, id_map):
    if self.is_backref:
        for field in self.fields:
            identifier = instance.__data__[field.name]
            key = (field, identifier)
            if key in id_map:
                setattr(instance, field.name, id_map[key])
    else:
        for field, attname in self.field_to_name:
            identifier = instance.__data__[field.rel_field.name]
            key = (field, identifier)
            rel_instances = id_map.get(key, [])
            for inst in rel_instances:
                setattr(inst, attname, instance)
                inst._dirty.clear()
            setattr(instance, field.backref, rel_instances)