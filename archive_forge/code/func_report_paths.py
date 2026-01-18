import atexit
import contextlib
import functools
import inspect
import io
import os
import platform
import sys
import threading
import traceback
import debugpy
from debugpy.common import json, timestamp, util
def report_paths(get_paths, label=None):
    prefix = f'    {label or get_paths}: '
    expr = None
    if not callable(get_paths):
        expr = get_paths
        get_paths = lambda: util.evaluate(expr)
    try:
        paths = get_paths()
    except AttributeError:
        report('{0}<missing>\n', prefix)
        return
    except Exception:
        swallow_exception('Error evaluating {0}', repr(expr) if expr else util.srcnameof(get_paths), level='info')
        return
    if not isinstance(paths, (list, tuple)):
        paths = [paths]
    for p in sorted(paths):
        report('{0}{1}', prefix, p)
        if p is not None:
            rp = os.path.realpath(p)
            if p != rp:
                report('({0})', rp)
        report('\n')
        prefix = ' ' * len(prefix)