from __future__ import absolute_import, division, print_function
import sys
import logging
import contextlib
import copy
import os
from future.utils import PY2, PY3
def disable_hooks():
    """
    Deprecated. Use remove_hooks() instead. This will be removed by
    ``future`` v1.0.
    """
    remove_hooks()