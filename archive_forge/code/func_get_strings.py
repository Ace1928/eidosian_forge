from __future__ import annotations
import datetime
import os
import threading
from abc import ABCMeta, abstractmethod
from asyncio import get_running_loop
from typing import AsyncGenerator, Iterable, Sequence
def get_strings(self) -> list[str]:
    """
        Get the strings from the history that are loaded so far.
        (In order. Oldest item first.)
        """
    return self._loaded_strings[::-1]