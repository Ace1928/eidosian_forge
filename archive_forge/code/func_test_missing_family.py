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
def test_missing_family(caplog):
    plt.rcParams['font.sans-serif'] = ['this-font-does-not-exist']
    with caplog.at_level('WARNING'):
        findfont('sans')
    assert [rec.getMessage() for rec in caplog.records] == ["findfont: Font family ['sans'] not found. Falling back to DejaVu Sans.", "findfont: Generic family 'sans' not found because none of the following families were found: this-font-does-not-exist"]