from __future__ import absolute_import, division, print_function
import sys
import logging
import contextlib
import copy
import os
from future.utils import PY2, PY3
def import_top_level_modules():
    with exclude_local_folder_imports(*TOP_LEVEL_MODULES):
        for m in TOP_LEVEL_MODULES:
            try:
                __import__(m)
            except ImportError:
                pass