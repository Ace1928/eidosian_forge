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
def set_database(self, database):
    self.database = database
    self.model._schema._database = database
    del self.table
    if isinstance(database, Proxy) and database.obj is None:
        database = None
    for hook in self._db_hooks:
        hook(database)