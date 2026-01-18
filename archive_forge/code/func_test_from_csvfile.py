import pytest
import rpy2.robjects as robjects
import rpy2.rlike.container as rlc
import array
import csv
import os
import tempfile
def test_from_csvfile():
    column_names = ('letter', 'value')
    data = (column_names, ('a', 1), ('b', 2), ('c', 3))
    fh = tempfile.NamedTemporaryFile(mode='w', delete=False)
    csv_w = csv.writer(fh)
    csv_w.writerows(data)
    fh.close()
    dataf = robjects.DataFrame.from_csvfile(fh.name)
    assert isinstance(dataf, robjects.DataFrame)
    assert column_names == tuple(dataf.names)
    assert dataf.nrow == 3
    assert dataf.ncol == 2