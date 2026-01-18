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
def patch_message(self, new_message):
    """
        Change the error message to the given new message.
        """
    self.args = (new_message,) + self.args[1:]