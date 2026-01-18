import ipywidgets as widgets
import pandas as pd
import numpy as np
import json
from types import FunctionType
from IPython.display import display
from numbers import Integral
from traitlets import (
from itertools import chain
from uuid import uuid4
from six import string_types
from distutils.version import LooseVersion
def should_be_stringified(col_series):
    return col_series.dtype == np.dtype('O') or hasattr(col_series, 'cat') or isinstance(col_series, pd.PeriodIndex)