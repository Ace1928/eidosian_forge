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
def get_selected_df(self):
    """
        Get a DataFrame which reflects the current state of the UI and only
        includes the currently selected row(s). Internally it calls
        ``get_changed_df()`` and then filters down to the selected rows
        using ``iloc``.

        :rtype: DataFrame
        """
    changed_df = self.get_changed_df()
    return changed_df.iloc[self._selected_rows]