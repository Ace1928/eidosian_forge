import numpy as np
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
import sys
import os
from importlib import import_module
from .tpot import TPOTClassifier, TPOTRegressor
from ._version import __version__
def float_range(value):
    """Ensure that the provided value is a float integer in the range [0., 1.].

    Parameters
    ----------
    value: float
        The number to evaluate

    Returns
    -------
    value: float
        Returns a float in the range (0., 1.)
    """
    try:
        value = float(value)
    except Exception:
        raise argparse.ArgumentTypeError("Invalid float value: '{}'".format(value))
    if value < 0.0 or value > 1.0:
        raise argparse.ArgumentTypeError("Invalid float value: '{}'".format(value))
    return value