import six
import sys
import time
import traceback
import random
import asyncio
import functools
def stop_after_attempt(self, previous_attempt_number, delay_since_first_attempt_ms):
    """Stop after the previous attempt >= stop_max_attempt_number."""
    return previous_attempt_number >= self._stop_max_attempt_number