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
@Node.copy
def for_update(self, for_update=True, of=None, nowait=None):
    if not for_update and (of is not None or nowait):
        for_update = True
    self._for_update = for_update
    self._for_update_of = of
    self._for_update_nowait = nowait