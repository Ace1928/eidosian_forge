import functools
import heapq
import logging
import random
import threading
import time
from collections import namedtuple
from itertools import chain
from peewee import MySQLDatabase
from peewee import PostgresqlDatabase
from peewee import SqliteDatabase
@locked
def close_stale(self, age=600):
    in_use = {}
    cutoff = time.time() - age
    n = 0
    for key, pool_conn in self._in_use.items():
        if pool_conn.checked_out < cutoff:
            self._close(pool_conn.connection, close_conn=True)
            n += 1
        else:
            in_use[key] = pool_conn
    self._in_use = in_use
    return n