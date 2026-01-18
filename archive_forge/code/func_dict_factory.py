from contextlib import contextmanager
import sqlite3
from eventlet import sleep
from eventlet import timeout
from oslo_log import log as logging
from glance.i18n import _LE
def dict_factory(cur, row):
    return {col[0]: row[idx] for idx, col in enumerate(cur.description)}