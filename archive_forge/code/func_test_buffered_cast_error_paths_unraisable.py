import sys
import pytest
import textwrap
import subprocess
import numpy as np
import numpy.core._multiarray_tests as _multiarray_tests
from numpy import array, arange, nditer, all
from numpy.testing import (
@pytest.mark.skipif(IS_WASM, reason='Cannot start subprocess')
@pytest.mark.skipif(not HAS_REFCOUNT, reason='PyPy seems to not hit this.')
def test_buffered_cast_error_paths_unraisable():
    code = textwrap.dedent('\n        import numpy as np\n    \n        it = np.nditer((np.array(1, dtype="i"),), op_dtypes=["S1"],\n                       op_flags=["writeonly"], casting="unsafe", flags=["buffered"])\n        buf = next(it)\n        buf[...] = "a"\n        del buf, it  # Flushing only happens during deallocate right now.\n        ')
    res = subprocess.check_output([sys.executable, '-c', code], stderr=subprocess.STDOUT, text=True)
    assert 'ValueError' in res