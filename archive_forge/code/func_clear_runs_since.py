import os
import hashlib
import pickle
import time
import shutil
import glob
from ..interfaces.base import BaseInterface
from ..pipeline.engine import Node
from ..pipeline.engine.utils import modify_paths
def clear_runs_since(self, day=None, month=None, year=None, warn=True):
    """Remove all the cache that where not used since the given date

        Parameters
        ==========
        day, month, year: integers, optional
            The integers specifying the latest day (in localtime) that
            a node should have been accessed to be kept. If not
            given, the current date is used.
        warn: boolean, optional
            If true, echoes warning messages for all directory
            removed
        """
    t = time.localtime()
    day = day if day is not None else t.tm_mday
    month = month if month is not None else t.tm_mon
    year = year if year is not None else t.tm_year
    base_dir = self.base_dir
    cut_off_file = '%s/log.%i/%02i/%02i.log' % (base_dir, year, month, day)
    logs_to_flush = list()
    recent_runs = dict()
    for log_name in glob.glob('%s/log.*/*/*.log' % base_dir):
        if log_name < cut_off_file:
            logs_to_flush.append(log_name)
        else:
            recent_runs = read_log(log_name, recent_runs)
    self._clear_all_but(recent_runs, warn=warn)
    for log_name in logs_to_flush:
        os.remove(log_name)