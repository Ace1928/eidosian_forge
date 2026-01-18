import inspect
import sys
from IPython.testing import decorators as dec
from IPython.testing.skipdoctest import skip_doctest
from IPython.utils.text import dedent
def call_doctest_bad():
    """Check that we can still call the decorated functions.
    
    >>> doctest_bad(3,y=4)
    x: 3
    y: 4
    k: {}
    """
    pass