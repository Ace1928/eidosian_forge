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
def change_selection(self, rows=[]):
    """
        Select a row (or rows) in the UI.  The indices of the
        rows to select are provided via the optional ``rows`` argument.

        Parameters
        ----------
        rows : list (default: [])
            A list of indices of the rows to select. For a multi-indexed
            DataFrame, each index in the list should be a tuple, with each
            value in each tuple corresponding to a level of the MultiIndex.
            The default value of ``[]`` results in the no rows being
            selected (i.e. it clears the selection).
        """
    new_selection = list(map(lambda x: self._df.index.get_loc(x), rows))
    self._change_selection(new_selection, 'api', send_msg_to_js=True)