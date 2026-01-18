import array
import pytest
import rpy2.rinterface_lib.callbacks
import rpy2.rinterface_lib.embedded
import rpy2.rlike.container as rlc
from rpy2 import robjects
from .. import utils
def test_extract_recycling_rule():
    v = robjects.vectors.IntVector(array.array('i', range(1, 23)))
    m = robjects.r.matrix(v, ncol=2)
    col = m.rx(True, 1)
    assert len(col) == 11