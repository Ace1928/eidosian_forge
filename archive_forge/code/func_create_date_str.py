from contextlib import ExitStack
import time
from typing import Type
from breezy import registry
from breezy import revision as _mod_revision
from breezy.osutils import format_date, local_time_offset
def create_date_str(timestamp=None, offset=None):
    """Just a wrapper around format_date to provide the right format.

    We don't want to use '%a' in the time string, because it is locale
    dependant. We also want to force timezone original, and show_offset

    Without parameters this function yields the current date in the local
    time zone.
    """
    if timestamp is None and offset is None:
        timestamp = time.time()
        offset = local_time_offset()
    return format_date(timestamp, offset, date_fmt='%Y-%m-%d %H:%M:%S', timezone='original', show_offset=True)