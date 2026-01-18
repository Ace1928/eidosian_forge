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
def test_donot_cache_tracebacks():

    class SomeObject:
        pass

    def inner():
        x = SomeObject()
        fig = mfigure.Figure()
        ax = fig.subplots()
        fig.text(0.5, 0.5, 'aardvark', family='doesnotexist')
        with BytesIO() as out:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')
                fig.savefig(out, format='raw')
    inner()
    for obj in gc.get_objects():
        if isinstance(obj, SomeObject):
            pytest.fail('object from inner stack still alive')