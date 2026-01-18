import inspect
import re
import sys
import textwrap
from pprint import pformat
from nltk.decorators import decorator  # this used in code that is commented out
from nltk.sem.logic import (
def foldemo(trace=None):
    """
    Interpretation of closed expressions in a first-order model.
    """
    folmodel(quiet=True)
    print()
    print('*' * mult)
    print('FOL Formulas Demo')
    print('*' * mult)
    formulas = ['love (adam, betty)', '(adam = mia)', '\\x. (boy(x) | girl(x))', '\\x. boy(x)(adam)', '\\x y. love(x, y)', '\\x y. love(x, y)(adam)(betty)', '\\x y. love(x, y)(adam, betty)', '\\x y. (boy(x) & love(x, y))', '\\x. exists y. (boy(x) & love(x, y))', 'exists z1. boy(z1)', 'exists x. (boy(x) &  -(x = adam))', 'exists x. (boy(x) & all y. love(y, x))', 'all x. (boy(x) | girl(x))', 'all x. (girl(x) -> exists y. boy(y) & love(x, y))', 'exists x. (boy(x) & all y. (girl(y) -> love(y, x)))', 'exists x. (boy(x) & all y. (girl(y) -> love(x, y)))', 'all x. (dog(x) -> - girl(x))', 'exists x. exists y. (love(x, y) & love(x, y))']
    for fmla in formulas:
        g2.purge()
        if trace:
            m2.evaluate(fmla, g2, trace)
        else:
            print(f"The value of '{fmla}' is: {m2.evaluate(fmla, g2)}")