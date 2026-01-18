import multiprocessing
import sys
import torch
from .reductions import init_reductions
from multiprocessing import *  # noqa: F403
from .spawn import (
def get_all_sharing_strategies():
    """Return a set of sharing strategies supported on a current system."""
    return _all_sharing_strategies