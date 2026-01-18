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
def close_idle(self):
    for _, conn in self._connections:
        self._close(conn, close_conn=True)
    self._connections = []