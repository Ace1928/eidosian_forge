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
def name_session(self, name):
    """Give the current session a name in the history database."""
    with self.db:
        self.db.execute('UPDATE sessions SET remark=? WHERE session==?', (name, self.session_number))