import re
import sys
import warnings
def check_string_type(value, title):
    if type(value) is str:
        return value
    raise AssertionError('{0} must be of type str (got {1})'.format(title, repr(value)))