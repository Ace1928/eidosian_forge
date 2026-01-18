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
def make_snake_case(s):
    first = SNAKE_CASE_STEP1.sub('\\1_\\2', s)
    return SNAKE_CASE_STEP2.sub('\\1_\\2', first).lower()