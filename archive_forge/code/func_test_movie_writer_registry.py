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
def test_movie_writer_registry():
    assert len(animation.writers._registered) > 0
    mpl.rcParams['animation.ffmpeg_path'] = 'not_available_ever_xxxx'
    assert not animation.writers.is_available('ffmpeg')
    bin = 'true' if sys.platform != 'win32' else 'where'
    mpl.rcParams['animation.ffmpeg_path'] = bin
    assert animation.writers.is_available('ffmpeg')