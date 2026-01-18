from itertools import groupby
import numpy as np
import pandas as pd
import param
from .dimension import Dimensioned, ViewableElement, asdim
from .layout import Composable, Layout, NdLayout
from .ndmapping import NdMapping
from .overlay import CompositeOverlay, NdOverlay, Overlayable
from .spaces import GridSpace, HoloMap
from .tree import AttrTree
from .util import get_param_values
def cell_type(self, row, col):
    """Type of the table cell, either 'data' or 'heading'

        Args:
            row (int): Integer index of table row
            col (int): Integer index of table column

        Returns:
            Type of the table cell, either 'data' or 'heading'
        """
    return 'heading' if row == 0 else 'data'