import time
import logging
import datetime
import threading
from pyzor.engines.common import Record, DBHandle, BaseEngine
def apply_method(self, method, varargs=(), kwargs=None):
    if kwargs is None:
        kwargs = {}
    with self.db_lock:
        return GdbmDBHandle.apply_method(self, method, varargs=varargs, kwargs=kwargs)