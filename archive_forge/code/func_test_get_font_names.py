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
@pytest.mark.skipif(sys.platform == 'win32', reason='Linux or OS only')
def test_get_font_names():
    paths_mpl = [cbook._get_data_path('fonts', subdir) for subdir in ['ttf']]
    fonts_mpl = findSystemFonts(paths_mpl, fontext='ttf')
    fonts_system = findSystemFonts(fontext='ttf')
    ttf_fonts = []
    for path in fonts_mpl + fonts_system:
        try:
            font = ft2font.FT2Font(path)
            prop = ttfFontProperty(font)
            ttf_fonts.append(prop.name)
        except Exception:
            pass
    available_fonts = sorted(list(set(ttf_fonts)))
    mpl_font_names = sorted(fontManager.get_font_names())
    assert set(available_fonts) == set(mpl_font_names)
    assert len(available_fonts) == len(mpl_font_names)
    assert available_fonts == mpl_font_names