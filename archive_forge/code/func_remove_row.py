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
def remove_row(self, rows=None):
    """
        Alias for ``remove_rows``, which is provided for convenience
        because this was the previous name of that method.
        """
    return self.remove_rows(rows)