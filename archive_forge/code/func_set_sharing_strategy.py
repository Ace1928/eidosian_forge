import multiprocessing
import sys
import torch
from .reductions import init_reductions
from multiprocessing import *  # noqa: F403
from .spawn import (
def set_sharing_strategy(new_strategy):
    """Set the strategy for sharing CPU tensors.

    Args:
        new_strategy (str): Name of the selected strategy. Should be one of
            the values returned by :func:`get_all_sharing_strategies()`.
    """
    global _sharing_strategy
    assert new_strategy in _all_sharing_strategies
    _sharing_strategy = new_strategy