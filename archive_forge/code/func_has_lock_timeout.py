import random
import threading
import time
import typing
from typing import Any
from typing import Mapping
import warnings
from ..api import CacheBackend
from ..api import NO_VALUE
from ... import util
def has_lock_timeout(self):
    return self.lock_timeout != 0