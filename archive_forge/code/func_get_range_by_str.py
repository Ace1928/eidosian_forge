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
def get_range_by_str(self, rangestr, raw=True, output=False):
    """Get lines of history from a string of ranges, as used by magic
        commands %hist, %save, %macro, etc.

        Parameters
        ----------
        rangestr : str
            A string specifying ranges, e.g. "5 ~2/1-4". If empty string is used,
            this will return everything from current session's history.

            See the documentation of :func:`%history` for the full details.

        raw, output : bool
            As :meth:`get_range`

        Returns
        -------
        Tuples as :meth:`get_range`
        """
    for sess, s, e in extract_hist_ranges(rangestr):
        for line in self.get_range(sess, s, e, raw=raw, output=output):
            yield line