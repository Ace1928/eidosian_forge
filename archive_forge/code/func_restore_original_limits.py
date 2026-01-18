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
def restore_original_limits(self):
    """Set the limits back to their original values"""
    for lib_controller, original_info in zip(self._controller.lib_controllers, self._original_info):
        lib_controller.set_num_threads(original_info['num_threads'])