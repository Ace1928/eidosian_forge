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
@decorator
def only_when_enabled(f, self, *a, **kw):
    """Decorator: return an empty list in the absence of sqlite."""
    if not self.enabled:
        return []
    else:
        return f(self, *a, **kw)