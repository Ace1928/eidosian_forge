from statsmodels.compat.python import lmap, lrange, lzip
import copy
from itertools import zip_longest
import time
import numpy as np
from statsmodels.iolib.table import SimpleTable
from statsmodels.iolib.tableformatting import (
from .summary2 import _model_types
def table_extend(tables, keep_headers=True):
    """extend a list of SimpleTables, adding titles to header of subtables

    This function returns the merged table as a deepcopy, in contrast to the
    SimpleTable extend method.

    Parameters
    ----------
    tables : list of SimpleTable instances
    keep_headers : bool
        If true, then all headers are kept. If falls, then the headers of
        subtables are blanked out.

    Returns
    -------
    table_all : SimpleTable
        merged tables as a single SimpleTable instance

    """
    from copy import deepcopy
    for ii, t in enumerate(tables[:]):
        t = deepcopy(t)
        if t[0].datatype == 'header':
            t[0][0].data = t.title
            t[0][0]._datatype = None
            t[0][0].row = t[0][1].row
            if not keep_headers and ii > 0:
                for c in t[0][1:]:
                    c.data = ''
        if ii == 0:
            table_all = t
        else:
            r1 = table_all[-1]
            r1.add_format('txt', row_dec_below='-')
            table_all.extend(t)
    table_all.title = None
    return table_all