import six
import sys
import time
import traceback
import random
import asyncio
import functools
def stop_after_delay(self, previous_attempt_number, delay_since_first_attempt_ms):
    """Stop after the time from the first attempt >= stop_max_delay."""
    return delay_since_first_attempt_ms >= self._stop_max_delay