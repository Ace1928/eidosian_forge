import base64
import functools
import itertools
import logging
import os
import queue
import random
import sys
import threading
import time
from types import TracebackType
from typing import (
import requests
import wandb
from wandb import util
from wandb.sdk.internal import internal_api
from ..lib import file_stream_utils
@staticmethod
def get_consecutive_offsets(console: Dict[int, str]) -> List[List[int]]:
    """Compress consecutive line numbers into an interval.

        Args:
            console: Dict[int, str] which maps offsets (line numbers) to lines of text.
            It represents a mini version of our console dashboard on the UI.

        Returns:
            A list of intervals (we compress consecutive line numbers into an interval).

        Example:
            >>> console = {2: "", 3: "", 4: "", 5: "", 10: "", 11: "", 20: ""}
            >>> get_consecutive_offsets(console)
            [(2, 5), (10, 11), (20, 20)]
        """
    offsets = sorted(list(console.keys()))
    intervals: List = []
    for i, num in enumerate(offsets):
        if i == 0:
            intervals.append([num, num])
            continue
        largest = intervals[-1][1]
        if num == largest + 1:
            intervals[-1][1] = num
        else:
            intervals.append([num, num])
    return intervals