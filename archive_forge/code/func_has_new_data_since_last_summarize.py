import contextlib
from datetime import datetime
import sys
import time
def has_new_data_since_last_summarize(self):
    return self._last_data_added_timestamp > self._last_summarized_timestamp