from statsmodels.compat.python import lzip
from io import StringIO
import numpy as np
from statsmodels.iolib import SimpleTable
def pprint_matrix(values, rlabels, clabels, col_space=None):
    buf = StringIO()
    T, K = (len(rlabels), len(clabels))
    if col_space is None:
        min_space = 10
        col_space = [max(len(str(c)) + 2, min_space) for c in clabels]
    else:
        col_space = (col_space,) * K
    row_space = max([len(str(x)) for x in rlabels]) + 2
    head = _pfixed('', row_space)
    for j, h in enumerate(clabels):
        head += _pfixed(h, col_space[j])
    buf.write(head + '\n')
    for i, rlab in enumerate(rlabels):
        line = ('%s' % rlab).ljust(row_space)
        for j in range(K):
            line += _pfixed(values[i, j], col_space[j])
        buf.write(line + '\n')
    return buf.getvalue()