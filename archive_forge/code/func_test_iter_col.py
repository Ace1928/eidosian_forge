import pytest
import rpy2.robjects as robjects
import rpy2.rlike.container as rlc
import array
import csv
import os
import tempfile
def test_iter_col():
    dataf = robjects.r('data.frame(a=1:2, b=I(c("a", "b")))')
    col_types = [x.typeof for x in dataf.iter_column()]
    assert rinterface.RTYPES.INTSXP == col_types[0]
    assert rinterface.RTYPES.STRSXP == col_types[1]