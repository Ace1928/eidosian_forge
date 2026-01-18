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
def manual_close(self):
    """
        Close the underlying connection without returning it to the pool.
        """
    if self.is_closed():
        return False
    conn = self.connection()
    self._in_use.pop(self.conn_key(conn), None)
    self.close()
    self._close(conn, close_conn=True)