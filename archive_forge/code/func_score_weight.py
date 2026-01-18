from base64 import b64encode
from collections import namedtuple
import copy
import dataclasses
from functools import lru_cache
from io import BytesIO
import json
import logging
from numbers import Number
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
from typing import Union
import matplotlib as mpl
from matplotlib import _api, _afm, cbook, ft2font
from matplotlib._fontconfig_pattern import (
from matplotlib.rcsetup import _validators
def score_weight(self, weight1, weight2):
    """
        Return a match score between *weight1* and *weight2*.

        The result is 0.0 if both weight1 and weight 2 are given as strings
        and have the same value.

        Otherwise, the result is the absolute value of the difference between
        the CSS numeric values of *weight1* and *weight2*, normalized between
        0.05 and 1.0.
        """
    if cbook._str_equal(weight1, weight2):
        return 0.0
    w1 = weight1 if isinstance(weight1, Number) else weight_dict[weight1]
    w2 = weight2 if isinstance(weight2, Number) else weight_dict[weight2]
    return 0.95 * (abs(w1 - w2) / 1000) + 0.05