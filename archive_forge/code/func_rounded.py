import collections
import datetime
import decimal
import numbers
import os
import os.path
import re
import time
from humanfriendly.compat import is_string, monotonic
from humanfriendly.deprecation import define_aliases
from humanfriendly.text import concatenate, format, pluralize, tokenize
@property
def rounded(self):
    """Human readable timespan rounded to seconds (a string)."""
    return format_timespan(round(self.elapsed_time))