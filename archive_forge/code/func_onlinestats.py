from __future__ import absolute_import, print_function, division
from collections import namedtuple
from petl.util.base import values, Table
def onlinestats(xi, n, mean=0, variance=0):
    meanprv = mean
    varianceprv = variance
    mean = ((n - 1) * meanprv + xi) / n
    variance = ((n - 1) * varianceprv + (xi - meanprv) * (xi - mean)) / n
    return (mean, variance)