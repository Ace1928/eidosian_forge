import inspect
import re
import sys
import textwrap
from pprint import pformat
from nltk.decorators import decorator  # this used in code that is commented out
from nltk.sem.logic import (
def satdemo(trace=None):
    """Satisfiers of an open formula in a first order model."""
    print()
    print('*' * mult)
    print('Satisfiers Demo')
    print('*' * mult)
    folmodel(quiet=True)
    formulas = ['boy(x)', '(x = x)', '(boy(x) | girl(x))', '(boy(x) & girl(x))', 'love(adam, x)', 'love(x, adam)', '-(x = adam)', 'exists z22. love(x, z22)', 'exists y. love(y, x)', 'all y. (girl(y) -> love(x, y))', 'all y. (girl(y) -> love(y, x))', 'all y. (girl(y) -> (boy(x) & love(y, x)))', '(boy(x) & all y. (girl(y) -> love(x, y)))', '(boy(x) & all y. (girl(y) -> love(y, x)))', '(boy(x) & exists y. (girl(y) & love(y, x)))', '(girl(x) -> dog(x))', 'all y. (dog(y) -> (x = y))', 'exists y. love(y, x)', 'exists y. (love(adam, y) & love(y, x))']
    if trace:
        print(m2)
    for fmla in formulas:
        print(fmla)
        Expression.fromstring(fmla)
    parsed = [Expression.fromstring(fmla) for fmla in formulas]
    for p in parsed:
        g2.purge()
        print("The satisfiers of '{}' are: {}".format(p, m2.satisfiers(p, 'x', g2, trace)))