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
def pprint_cell(self, row, col):
    """Formatted contents of table cell.

        Args:
            row (int): Integer index of table row
            col (int): Integer index of table column

        Returns:
            Formatted table cell contents
        """
    ndims = self.ndims
    if col >= self.cols:
        raise Exception('Maximum column index is %d' % self.cols - 1)
    elif row >= self.rows:
        raise Exception('Maximum row index is %d' % self.rows - 1)
    elif row == 0:
        if col >= ndims:
            if self.vdims:
                return self.vdims[col - ndims].pprint_label
            else:
                return ''
        return self.kdims[col].pprint_label
    else:
        dim = self.get_dimension(col)
        return dim.pprint_value(self.iloc[row - 1, col])