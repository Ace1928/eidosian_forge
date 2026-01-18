import array
import pytest
import rpy2.rinterface_lib.callbacks
import rpy2.rinterface_lib.embedded
import rpy2.rlike.container as rlc
from rpy2 import robjects
from .. import utils
def test_extract_by_index():
    seq_R = robjects.baseenv['seq']
    mySeq = seq_R(0, 10)
    myIndex = robjects.vectors.IntVector(array.array('i', range(1, 11, 2)))
    mySubset = mySeq.rx(myIndex, drop=True)
    for i, si in enumerate(myIndex):
        assert mySeq[si - 1] == mySubset[i]