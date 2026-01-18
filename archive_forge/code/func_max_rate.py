import threading
import time
from botocore.exceptions import CapacityNotAvailableError
@max_rate.setter
def max_rate(self, value):
    with self._new_fill_rate_condition:
        self._refill()
        self._fill_rate = max(value, self._min_rate)
        if value >= 1:
            self._max_capacity = value
        else:
            self._max_capacity = 1
        self._current_capacity = min(self._current_capacity, self._max_capacity)
        self._new_fill_rate_condition.notify()