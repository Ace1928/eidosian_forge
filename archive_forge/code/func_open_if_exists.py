import enum
import json
import os
import re
import typing as t
from collections import abc
from collections import deque
from random import choice
from random import randrange
from threading import Lock
from types import CodeType
from urllib.parse import quote_from_bytes
import markupsafe
def open_if_exists(filename: str, mode: str='rb') -> t.Optional[t.IO]:
    """Returns a file descriptor for the filename if that file exists,
    otherwise ``None``.
    """
    if not os.path.isfile(filename):
        return None
    return open(filename, mode)