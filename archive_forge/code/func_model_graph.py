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
def model_graph(self, refs=True, backrefs=True, depth_first=True):
    if not refs and (not backrefs):
        raise ValueError('One of `refs` or `backrefs` must be True.')
    accum = [(None, self.model, None)]
    seen = set()
    queue = collections.deque((self,))
    method = queue.pop if depth_first else queue.popleft
    while queue:
        curr = method()
        if curr in seen:
            continue
        seen.add(curr)
        if refs:
            for fk, model in curr.refs.items():
                accum.append((fk, model, False))
                queue.append(model._meta)
        if backrefs:
            for fk, model in curr.backrefs.items():
                accum.append((fk, model, True))
                queue.append(model._meta)
    return accum