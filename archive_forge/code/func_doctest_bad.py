import inspect
import sys
from IPython.testing import decorators as dec
from IPython.testing.skipdoctest import skip_doctest
from IPython.utils.text import dedent
@skip_doctest
def doctest_bad(x, y=1, **k):
    """A function whose doctest we need to skip.

    >>> 1+1
    3
    """
    print('x:', x)
    print('y:', y)
    print('k:', k)