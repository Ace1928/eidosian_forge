import os
import re
import sys
import ctypes
import textwrap
from typing import final
import warnings
from ctypes.util import find_library
from abc import ABC, abstractmethod
from functools import lru_cache
from contextlib import ContextDecorator
@property
def loaded_backends(self):
    return self._get_backend_list(loaded=True)