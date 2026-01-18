import itertools
import math
import operator
import random
from functools import reduce
def makeFakeProps():
    mw = random.randint(10, 500)
    alogp = random.randint(-10, 10)
    tpsa = random.randint(0, 180)
    return [mw, alogp, tpsa]