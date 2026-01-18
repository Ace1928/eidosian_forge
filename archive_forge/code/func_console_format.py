import argparse
import collections
import functools
import sys
import time
from oslo_serialization import jsonutils
from oslo_utils import importutils
from oslo_log import log
def console_format(prefix, locator, record, loggers=[], levels=[], level_key=DEFAULT_LEVEL_KEY, traceback_key=DEFAULT_TRACEBACK_KEY):
    record = collections.defaultdict(str, record)
    if loggers:
        name = record.get('name')
        if not any((name.startswith(n) for n in loggers)):
            return
    if levels:
        if record.get(level_key) not in levels:
            return
    levelname = record.get(level_key)
    if levelname:
        record[level_key] = colorise(levelname)
    try:
        prefix = prefix % record
    except TypeError:
        yield warn('Missing non-string placeholder in record', {str(k): str(v) if isinstance(v, str) else v for k, v in record.items()})
        return
    locator = ''
    if record.get('levelno', 100) <= log.DEBUG or levelname == 'DEBUG':
        locator = locator % record
    yield ' '.join((x for x in [prefix, record['message'], locator] if x))
    tb = record.get(traceback_key)
    if tb:
        if type(tb) is str:
            tb = tb.rstrip().split('\n')
        for tb_line in tb:
            yield ' '.join([prefix, tb_line])