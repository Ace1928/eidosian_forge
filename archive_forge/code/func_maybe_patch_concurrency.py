import os
import re
import sys
from collections import namedtuple
from . import local
def maybe_patch_concurrency(argv=None, short_opts=None, long_opts=None, patches=None):
    """Apply eventlet/gevent monkeypatches.

    With short and long opt alternatives that specify the command line
    option to set the pool, this makes sure that anything that needs
    to be patched is completed as early as possible.
    (e.g., eventlet/gevent monkey patches).
    """
    argv = argv if argv else sys.argv
    short_opts = short_opts if short_opts else ['-P']
    long_opts = long_opts if long_opts else ['--pool']
    patches = patches if patches else {'eventlet': _patch_eventlet, 'gevent': _patch_gevent}
    try:
        pool = _find_option_with_arg(argv, short_opts, long_opts)
    except KeyError:
        pass
    else:
        try:
            patcher = patches[pool]
        except KeyError:
            pass
        else:
            patcher()
        from celery import concurrency
        if pool in concurrency.get_available_pool_names():
            concurrency.get_implementation(pool)