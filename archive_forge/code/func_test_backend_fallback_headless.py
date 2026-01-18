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
@pytest.mark.skipif(sys.platform != 'linux', reason='Linux only')
def test_backend_fallback_headless(tmpdir):
    env = {**os.environ, 'DISPLAY': '', 'WAYLAND_DISPLAY': '', 'MPLBACKEND': '', 'MPLCONFIGDIR': str(tmpdir)}
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run([sys.executable, '-c', "import matplotlib;matplotlib.use('tkagg');import matplotlib.pyplot;matplotlib.pyplot.plot(42);"], env=env, check=True, stderr=subprocess.DEVNULL)