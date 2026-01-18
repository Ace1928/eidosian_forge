import csv
import functools
import itertools
import math
import os
import re
from io import BytesIO
from pathlib import Path
from string import ascii_letters
from typing import Union
import numpy as np
import pandas
import psutil
import pytest
from pandas.core.dtypes.common import (
import modin.pandas as pd
from modin.config import (
from modin.pandas.io import to_pandas
from modin.pandas.testing import (
from modin.utils import try_cast_to_pandas
def assert_all_act_same(condition, *objs):
    """
    Assert that all of the objs give the same boolean result for the passed condition (either all True or all False).

    Parameters
    ----------
    condition : callable(obj) -> bool
        Condition to run on the passed objects.
    *objs :
        Objects to pass to the condition.

    Returns
    -------
    bool
        Result of the condition.
    """
    results = [condition(obj) for obj in objs]
    if len(results) < 2:
        return results[0] if len(results) else None
    assert all((results[0] == res for res in results[1:]))
    return results[0]