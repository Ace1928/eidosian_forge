import gc
import os
import warnings
import threading
import contextlib
from abc import ABCMeta, abstractmethod
from ._multiprocessing_helpers import mp
def get_exceptions(self):
    """List of exception types to be captured."""
    return []