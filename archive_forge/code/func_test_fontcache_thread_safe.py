from io import BytesIO, StringIO
import gc
import multiprocessing
import os
from pathlib import Path
from PIL import Image
import shutil
import subprocess
import sys
import warnings
import numpy as np
import pytest
from matplotlib.font_manager import (
from matplotlib import cbook, ft2font, pyplot as plt, rc_context, figure as mfigure
def test_fontcache_thread_safe():
    pytest.importorskip('threading')
    import inspect
    proc = subprocess.run([sys.executable, '-c', inspect.getsource(_test_threading) + '\n_test_threading()'])
    if proc.returncode:
        pytest.fail(f'The subprocess returned with non-zero exit status {proc.returncode}.')