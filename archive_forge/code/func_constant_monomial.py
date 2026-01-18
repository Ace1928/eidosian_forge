import re
import operator
from fractions import Fraction
import sys
@classmethod
def constant_monomial(cls, coefficient):
    return Monomial(coefficient, ())