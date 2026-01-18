import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import weakref
import numpy as np
import pytest
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.testing.decorators import check_figures_equal
def test_movie_writer_invalid_path(anim):
    if sys.platform == 'win32':
        match_str = "\\[WinError 3] .*'\\\\\\\\foo\\\\\\\\bar\\\\\\\\aardvark'"
    else:
        match_str = "\\[Errno 2] .*'/foo"
    with pytest.raises(FileNotFoundError, match=match_str):
        anim.save('/foo/bar/aardvark/thiscannotreallyexist.mp4', writer=animation.FFMpegFileWriter())