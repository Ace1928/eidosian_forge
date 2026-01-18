import abc
import time
import warnings
from collections import namedtuple
from functools import wraps
from typing import Dict, Optional
def get_elapsed_time_ms(start_time_in_seconds: float):
    """Return the elapsed time in millis from the given start time."""
    end_time = time.time()
    return int((end_time - start_time_in_seconds) * 1000)