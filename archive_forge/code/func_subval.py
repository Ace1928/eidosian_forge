import operator
import sys
import warnings
def subval(x, i, v):
    x_ = list(x)
    x_[i] = v
    return tuple(x_)