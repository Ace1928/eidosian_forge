import gc
import signal
import sys
import textwrap
import weakref
import pytest
from pyarrow.util import doc, _break_traceback_cycle_from_frame
from pyarrow.tests.util import disabled_gc
@doc(func_a, textwrap.dedent('\n        Examples\n        --------\n\n        >>> func_b()\n        B\n        '), method='func_b', operation='B')
def func_b(whatever):
    pass