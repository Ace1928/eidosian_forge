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
def store_inputs(self, line_num, source, source_raw=None):
    """Store source and raw input in history and create input cache
        variables ``_i*``.

        Parameters
        ----------
        line_num : int
            The prompt number of this input.
        source : str
            Python input.
        source_raw : str, optional
            If given, this is the raw input without any IPython transformations
            applied to it.  If not given, ``source`` is used.
        """
    if source_raw is None:
        source_raw = source
    source = source.rstrip('\n')
    source_raw = source_raw.rstrip('\n')
    if self._exit_re.match(source_raw.strip()):
        return
    self.input_hist_parsed.append(source)
    self.input_hist_raw.append(source_raw)
    with self.db_input_cache_lock:
        self.db_input_cache.append((line_num, source, source_raw))
        if len(self.db_input_cache) >= self.db_cache_size:
            self.save_flag.set()
    self._iii = self._ii
    self._ii = self._i
    self._i = self._i00
    self._i00 = source_raw
    new_i = '_i%s' % line_num
    to_main = {'_i': self._i, '_ii': self._ii, '_iii': self._iii, new_i: self._i00}
    if self.shell is not None:
        self.shell.push(to_main, interactive=False)