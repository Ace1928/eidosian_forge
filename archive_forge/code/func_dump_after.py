import os
import sys
import time
import threading
import traceback
from debugpy.common import log
def dump_after(secs):
    """Invokes dump() on a background thread after waiting for the specified time."""

    def dumper():
        time.sleep(secs)
        try:
            dump()
        except:
            log.swallow_exception()
    thread = threading.Thread(target=dumper)
    thread.daemon = True
    thread.start()