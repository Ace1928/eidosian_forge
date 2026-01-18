import functools
import os
import sys
import re
import shutil
import types
from .encoding import DEFAULT_ENCODING
import platform
def doctest_refactor_print(func_or_str):
    return func_or_str