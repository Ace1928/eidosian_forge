import re
import operator
from fractions import Fraction
import sys
def default_print_coefficient_method(i):
    try:
        sign = '+' if i >= 0 else '-'
        if abs(i) == 1:
            print_str = None
        else:
            print_str = str(abs(i))
        return (sign, print_str)
    except (TypeError, ValueError):
        return uncomparable_print_coefficient_method(i)