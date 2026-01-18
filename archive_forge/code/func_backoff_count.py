import random
import time
import six
@property
def backoff_count(self):
    """The current amount of backoff attempts that have been made."""
    return self._backoff_count