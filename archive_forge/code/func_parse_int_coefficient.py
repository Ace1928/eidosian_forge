import re
import operator
from fractions import Fraction
import sys
def parse_int_coefficient(s):
    coeff, rest = re.match('([0-9]*)(.*)', s).groups()
    if coeff:
        coeff = int(coeff)
    else:
        coeff = None
    return (coeff, rest)