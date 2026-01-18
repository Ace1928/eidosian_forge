import abc
import contextlib
import os
import sys
import warnings
import numba.core.config
import numpy as np
from collections import defaultdict
from functools import wraps
from abc import abstractmethod
def termcolor():
    global _termcolor_inst
    if _termcolor_inst is None:
        _termcolor_inst = NOPColorScheme()
    return _termcolor_inst