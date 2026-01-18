from statsmodels.compat.python import lmap, lrange, lzip
import copy
from itertools import zip_longest
import time
import numpy as np
from statsmodels.iolib.table import SimpleTable
from statsmodels.iolib.tableformatting import (
from .summary2 import _model_types
def summary_return(tables, return_fmt='text'):
    if return_fmt == 'text':
        strdrop = lambda x: str(x).rsplit('\n', 1)[0]
        return '\n'.join(lmap(strdrop, tables[:-1]) + [str(tables[-1])])
    elif return_fmt == 'tables':
        return tables
    elif return_fmt == 'csv':
        return '\n'.join((x.as_csv() for x in tables))
    elif return_fmt == 'latex':
        table = copy.deepcopy(tables[0])
        for part in tables[1:]:
            table.extend(part)
        return table.as_latex_tabular()
    elif return_fmt == 'html':
        return '\n'.join((table.as_html() for table in tables))
    else:
        raise ValueError('available output formats are text, csv, latex, html')