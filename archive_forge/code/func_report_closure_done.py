import faulthandler
import os
import sys
import threading
import time
from absl import logging
def report_closure_done(self):
    if self._timeout > 0:
        self._last_activity_time = time.time()