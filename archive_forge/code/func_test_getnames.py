import pytest
import rpy2.robjects as robjects
import os
import array
import time
import datetime
import rpy2.rlike.container as rlc
from collections import OrderedDict
def test_getnames():
    vec = robjects.vectors.IntVector(array.array('i', [1, 2, 3]))
    v_names = [robjects.baseenv['letters'][x] for x in (0, 1, 2)]
    r_names = robjects.baseenv['c'](*v_names)
    vec = robjects.baseenv['names<-'](vec, r_names)
    for i in range(len(vec)):
        assert v_names[i] == vec.names[i]
    vec.names[0] = 'x'