import abc
import logging
import threading
import time
from contextlib import contextmanager
from inspect import getframeinfo, stack
from typing import Any, Dict, List, Optional, Set
@abc.abstractmethod
def clear_timers(self, worker_ids: Set[Any]) -> None:
    """
        Clears all timers for the given ``worker_ids``.
        """
    pass