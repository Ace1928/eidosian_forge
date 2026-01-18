import datetime
import decimal
import functools
import logging
import time
import warnings
from contextlib import contextmanager
from hashlib import md5
from django.apps import apps
from django.db import NotSupportedError
from django.utils.dateparse import parse_time
@contextmanager
def debug_sql(self, sql=None, params=None, use_last_executed_query=False, many=False):
    start = time.monotonic()
    try:
        yield
    finally:
        stop = time.monotonic()
        duration = stop - start
        if use_last_executed_query:
            sql = self.db.ops.last_executed_query(self.cursor, sql, params)
        try:
            times = len(params) if many else ''
        except TypeError:
            times = '?'
        self.db.queries_log.append({'sql': '%s times: %s' % (times, sql) if many else sql, 'time': '%.3f' % duration})
        logger.debug('(%.3f) %s; args=%s; alias=%s', duration, sql, params, self.db.alias, extra={'duration': duration, 'sql': sql, 'params': params, 'alias': self.db.alias})