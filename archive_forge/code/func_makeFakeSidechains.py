import itertools
import math
import operator
import random
from functools import reduce
def makeFakeSidechains(lib, num):
    res = []
    for i in range(num):
        res.append(Sidechain(lib + '_' + str(i), makeFakeProps()))
    return res