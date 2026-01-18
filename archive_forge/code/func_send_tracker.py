import contextlib
from datetime import datetime
import sys
import time
@contextlib.contextmanager
def send_tracker(self):
    """Create a context manager for a round of data sending."""
    self._send_count += 1
    if self._send_count == 1:
        self._single_line_message('Started scanning logdir.')
    try:
        self._overwrite_line_message('Data upload starting')
        yield
    finally:
        self._update_cumulative_status()
        if self._one_shot:
            self._single_line_message('Done scanning logdir.')
        else:
            self._overwrite_line_message('Listening for new data in logdir', color_code=_STYLE_YELLOW)