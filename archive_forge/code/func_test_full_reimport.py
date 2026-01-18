from numpy.testing import (
from numpy.compat import pickle
import pytest
import sys
import subprocess
import textwrap
from importlib import reload
@pytest.mark.skipif(IS_WASM, reason="can't start subprocess")
def test_full_reimport():
    """At the time of writing this, it is *not* truly supported, but
    apparently enough users rely on it, for it to be an annoying change
    when it started failing previously.
    """
    code = textwrap.dedent('\n        import sys\n        from pytest import warns\n        import numpy as np\n\n        for k in list(sys.modules.keys()):\n            if "numpy" in k:\n                del sys.modules[k]\n\n        with warns(UserWarning):\n            import numpy as np\n        ')
    p = subprocess.run([sys.executable, '-c', code], capture_output=True)
    if p.returncode:
        raise AssertionError(f'Non-zero return code: {p.returncode!r}\n\n{p.stderr.decode()}')