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
def ensure_join(self, lm, rm, on=None, **join_kwargs):
    join_ctx = self._join_ctx
    for dest, _, constructor, _ in self._joins.get(lm, []):
        if dest == rm:
            return self
    return self.switch(lm).join(rm, on=on, **join_kwargs).switch(join_ctx)