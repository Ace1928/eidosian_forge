import codecs
import datetime
import functools
from io import BytesIO
import logging
import math
import os
import pathlib
import shutil
import subprocess
from tempfile import TemporaryDirectory
import weakref
from PIL import Image
import matplotlib as mpl
from matplotlib import _api, cbook, font_manager as fm
from matplotlib.backend_bases import (
from matplotlib.backends.backend_mixed import MixedModeRenderer
from matplotlib.backends.backend_pdf import (
from matplotlib.path import Path
from matplotlib.figure import Figure
from matplotlib._pylab_helpers import Gcf
def get_width_height_descent(self, text, prop):
    """
        Get the width, total height, and descent (in TeX points) for a text
        typeset by the current LaTeX environment.
        """
    return self._get_box_metrics(_escape_and_apply_props(text, prop))