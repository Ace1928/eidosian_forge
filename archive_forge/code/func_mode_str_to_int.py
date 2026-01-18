from abc import ABC, abstractmethod
import calendar
from collections import deque
from datetime import datetime, timedelta, tzinfo
from string import digits
import re
import time
import warnings
from git.util import IterableList, IterableObj, Actor
from typing import (
from git.types import Has_id_attribute, Literal  # , _T
def mode_str_to_int(modestr: Union[bytes, str]) -> int:
    """
    :param modestr:
        String like 755 or 644 or 100644 - only the last 6 chars will be used.

    :return:
        String identifying a mode compatible to the mode methods ids of the
        stat module regarding the rwx permissions for user, group and other,
        special flags and file system flags, such as whether it is a symlink.
    """
    mode = 0
    for iteration, char in enumerate(reversed(modestr[-6:])):
        char = cast(Union[str, int], char)
        mode += int(char) << iteration * 3
    return mode