from datetime import timedelta
import os
import pickle
import platform as pl
import sys
import numpy as np
import pandas
from pandas import (
from pandas.arrays import SparseArray
from pandas.tseries.offsets import (
def platform_name():
    return '_'.join([str(pandas.__version__), str(pl.machine()), str(pl.system().lower()), str(pl.python_version())])