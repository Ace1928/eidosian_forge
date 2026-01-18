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
@catch_corrupt_db
def get_last_session_id(self):
    """Get the last session ID currently in the database.

        Within IPython, this should be the same as the value stored in
        :attr:`HistoryManager.session_number`.
        """
    for record in self.get_tail(n=1, include_latest=True):
        return record[0]