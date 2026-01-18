import collections
import os
import pickle
from abc import ABC
from typing import (
import numpy
from . import collective
from .core import Booster, DMatrix, XGBoostError, _parse_eval_str
def get_s(value: _Score) -> float:
    """get score if it's cross validation history."""
    return value[0] if isinstance(value, tuple) else value