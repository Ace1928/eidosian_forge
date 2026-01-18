from mpmath.libmp import *
from mpmath import *
import random
import time
import math
import cmath
def test_misc_bugs():
    mp.dps = 1000
    log(1302)
    mp.dps = 15