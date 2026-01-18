import pytest
import rpy2.robjects as robjects
import rpy2.rlike.container as rlc
import array
import csv
import os
import tempfile
def test_init_stringsasfactors():
    od = {'a': robjects.IntVector((1, 2)), 'b': robjects.StrVector(('c', 'd'))}
    dataf = robjects.DataFrame(od, stringsasfactor=True)
    assert isinstance(dataf.rx2('b'), robjects.FactorVector)
    dataf = robjects.DataFrame(od, stringsasfactor=False)
    assert isinstance(dataf.rx2('b'), robjects.StrVector)