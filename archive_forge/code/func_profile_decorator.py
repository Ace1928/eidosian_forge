import sys
import os
import hotshot
import hotshot.stats
import threading
import cgi
import time
from io import StringIO
from paste import response
def profile_decorator(**options):
    """
    Profile a single function call.

    Used around a function, like::

        @profile_decorator(options...)
        def ...

    All calls to the function will be profiled.  The options are
    all keywords, and are:

        log_file:
            The filename to log to (or ``'stdout'`` or ``'stderr'``).
            Default: stderr.
        display_limit:
            Only show the top N items, default: 20.
        sort_stats:
            A list of string-attributes to sort on.  Default
            ``('time', 'calls')``.
        strip_dirs:
            Strip directories/module names from files?  Default True.
        add_info:
            If given, this info will be added to the report (for your
            own tracking).  Default: none.
        log_filename:
            The temporary filename to log profiling data to.  Default;
            ``./profile_data.log.tmp``
        no_profile:
            If true, then don't actually profile anything.  Useful for
            conditional profiling.
    """
    if options.get('no_profile'):

        def decorator(func):
            return func
        return decorator

    def decorator(func):

        def replacement(*args, **kw):
            return DecoratedProfile(func, **options)(*args, **kw)
        return replacement
    return decorator