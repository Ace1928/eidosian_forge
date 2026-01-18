import gc
import signal
import sys
import textwrap
import weakref
import pytest
from pyarrow.util import doc, _break_traceback_cycle_from_frame
from pyarrow.tests.util import disabled_gc
def test_docstring_append():
    docstr = textwrap.dedent('\n        This is the func_c method.\n\n        It computes C.\n\n        Examples\n        --------\n\n        >>> func_c()\n        C\n        ')
    assert func_c.__doc__ == docstr