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
def test_json_serialization(tmpdir):
    path = Path(tmpdir, 'fontlist.json')
    json_dump(fontManager, path)
    copy = json_load(path)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'findfont: Font family.*not found')
        for prop in ({'family': 'STIXGeneral'}, {'family': 'Bitstream Vera Sans', 'weight': 700}, {'family': 'no such font family'}):
            fp = FontProperties(**prop)
            assert fontManager.findfont(fp, rebuild_if_missing=False) == copy.findfont(fp, rebuild_if_missing=False)