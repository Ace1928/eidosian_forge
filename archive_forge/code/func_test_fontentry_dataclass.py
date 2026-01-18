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
def test_fontentry_dataclass():
    fontent = FontEntry(name='font-name')
    png = fontent._repr_png_()
    img = Image.open(BytesIO(png))
    assert img.width > 0
    assert img.height > 0
    html = fontent._repr_html_()
    assert html.startswith('<img src="data:image/png;base64')