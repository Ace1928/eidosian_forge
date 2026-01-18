import functools
import re
def add_space_between_words(s, c):
    if len(s) > 0 and s[-1].islower() and c.isupper():
        return s + ' ' + c
    return s + c