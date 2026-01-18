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
def get_changed_df(self):
    """
        Get a copy of the DataFrame that was used to create the current
        instance of QgridWidget which reflects the current state of the UI.
        This includes any sorting or filtering changes, as well as edits
        that have been made by double clicking cells.

        :rtype: DataFrame
        """
    col_names_to_drop = list(self._sort_helper_columns.values())
    col_names_to_drop.append(self._index_col_name)
    return self._df.drop(col_names_to_drop, axis=1)