import gc
import signal
import sys
import textwrap
import weakref
import pytest
from pyarrow.util import doc, _break_traceback_cycle_from_frame
from pyarrow.tests.util import disabled_gc
@doc(func_a, method='func_c', operation='C')
def func_c(whatever):
    """
    Examples
    --------

    >>> func_c()
    C
    """
    pass