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
@pytest.mark.skipif(sys.platform == 'win32' or not has_fclist, reason='no fontconfig installed')
def test_get_fontconfig_fonts():
    assert len(_get_fontconfig_fonts()) > 1