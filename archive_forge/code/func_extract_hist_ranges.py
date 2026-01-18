import atexit
import datetime
import re
import sqlite3
import threading
from pathlib import Path
from decorator import decorator
from traitlets import (
from traitlets.config.configurable import LoggingConfigurable
from IPython.paths import locate_profile
from IPython.utils.decorators import undoc
def extract_hist_ranges(ranges_str):
    """Turn a string of history ranges into 3-tuples of (session, start, stop).

    Empty string results in a `[(0, 1, None)]`, i.e. "everything from current
    session".

    Examples
    --------
    >>> list(extract_hist_ranges("~8/5-~7/4 2"))
    [(-8, 5, None), (-7, 1, 5), (0, 2, 3)]
    """
    if ranges_str == '':
        yield (0, 1, None)
        return
    for range_str in ranges_str.split():
        rmatch = range_re.match(range_str)
        if not rmatch:
            continue
        start = rmatch.group('start')
        if start:
            start = int(start)
            end = rmatch.group('end')
            end = int(end) if end else start + 1
        else:
            if not rmatch.group('startsess'):
                continue
            start = 1
            end = None
        if rmatch.group('sep') == '-':
            end += 1
        startsess = rmatch.group('startsess') or '0'
        endsess = rmatch.group('endsess') or startsess
        startsess = int(startsess.replace('~', '-'))
        endsess = int(endsess.replace('~', '-'))
        assert endsess >= startsess, 'start session must be earlier than end session'
        if endsess == startsess:
            yield (startsess, start, end)
            continue
        yield (startsess, start, None)
        for sess in range(startsess + 1, endsess):
            yield (sess, 1, None)
        yield (endsess, 1, end)