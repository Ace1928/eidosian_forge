from __future__ import annotations
import contextlib
import string
import warnings
import numpy as np
import pandas as pd
from packaging.version import Version
import pandas.testing as tm
def makeTimeDataFrame():
    data = makeDataFrame()
    data.index = makeDateIndex()
    return data