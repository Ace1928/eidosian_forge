import operator
import math
import cmath
def rgamma(x):
    try:
        return 1.0 / gamma(x)
    except ZeroDivisionError:
        return x * 0.0