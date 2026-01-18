import os
import re
import time
from os.path import basename
from multiprocessing import util
def concurrency_safe_rename(src, dst):
    """Renames ``src`` into ``dst`` overwriting ``dst`` if it exists.

        On Windows os.replace can yield permission errors if executed by two
        different processes.
        """
    max_sleep_time = 1
    total_sleep_time = 0
    sleep_time = 0.001
    while total_sleep_time < max_sleep_time:
        try:
            replace(src, dst)
            break
        except Exception as exc:
            if getattr(exc, 'winerror', None) in access_denied_errors:
                time.sleep(sleep_time)
                total_sleep_time += sleep_time
                sleep_time *= 2
            else:
                raise
    else:
        raise