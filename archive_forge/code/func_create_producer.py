from __future__ import annotations
import os
from itertools import chain
from .connection import Resource
from .messaging import Producer
from .utils.collections import EqualityDict
from .utils.compat import register_after_fork
from .utils.functional import lazy
def create_producer(self):
    conn = self._acquire_connection()
    try:
        return self.Producer(conn)
    except BaseException:
        conn.release()
        raise