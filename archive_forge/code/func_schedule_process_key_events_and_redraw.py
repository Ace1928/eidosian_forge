from .hyperboloid_utilities import *
import time
import sys
import tempfile
import png
def schedule_process_key_events_and_redraw(self, time_ms):
    """
        Schedule call to process_key_events_and_redraw in given time
        (milliseconds) if not scheduled already.
        """
    if self.process_keys_and_redraw_scheduled:
        return
    self.process_keys_and_redraw_scheduled = True
    self.after(time_ms, self.process_key_events_and_redraw)