import gc
import os
import warnings
import threading
import contextlib
from abc import ABCMeta, abstractmethod
from ._multiprocessing_helpers import mp
def reset_batch_stats(self):
    """Reset batch statistics to default values.

        This avoids interferences with future jobs.
        """
    self._effective_batch_size = self._DEFAULT_EFFECTIVE_BATCH_SIZE
    self._smoothed_batch_duration = self._DEFAULT_SMOOTHED_BATCH_DURATION