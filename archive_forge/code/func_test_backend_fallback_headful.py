import copy
import os
from pathlib import Path
import subprocess
import sys
from unittest import mock
from cycler import cycler, Cycler
import pytest
import matplotlib as mpl
from matplotlib import _api, _c_internal_utils
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from matplotlib.rcsetup import (
@pytest.mark.skipif(sys.platform == 'linux' and (not _c_internal_utils.display_is_valid()), reason='headless')
def test_backend_fallback_headful(tmpdir):
    pytest.importorskip('tkinter')
    env = {**os.environ, 'MPLBACKEND': '', 'MPLCONFIGDIR': str(tmpdir)}
    backend = subprocess.check_output([sys.executable, '-c', "import matplotlib as mpl; sentinel = mpl.rcsetup._auto_backend_sentinel; assert mpl.RcParams({'backend': sentinel})['backend'] == sentinel; assert mpl.rcParams._get('backend') == sentinel; import matplotlib.pyplot; print(matplotlib.get_backend())"], env=env, text=True)
    assert backend.strip().lower() != 'agg'