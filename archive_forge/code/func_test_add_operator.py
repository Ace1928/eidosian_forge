import pytest
import rpy2.robjects as robjects
import os
import array
import time
import datetime
import rpy2.rlike.container as rlc
from collections import OrderedDict
def test_add_operator():
    seq_R = robjects.r['seq']
    mySeqA = seq_R(0, 3)
    mySeqB = seq_R(5, 7)
    mySeqAdd = mySeqA + mySeqB
    assert len(mySeqA) + len(mySeqB) == len(mySeqAdd)
    for i, li in enumerate(mySeqA):
        assert mySeqAdd[i] == li
    for j, li in enumerate(mySeqB):
        assert mySeqAdd[i + j + 1] == li