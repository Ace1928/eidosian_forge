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
def test_animation_frame_formats():
    for fmt in ['png', 'jpeg', 'tiff', 'raw', 'rgba', 'ppm', 'sgi', 'bmp', 'pbm', 'svg']:
        mpl.rcParams['animation.frame_format'] = fmt