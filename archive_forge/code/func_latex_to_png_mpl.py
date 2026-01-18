from io import BytesIO, open
import os
import tempfile
import shutil
import subprocess
from base64 import encodebytes
import textwrap
from pathlib import Path
from IPython.utils.process import find_cmd, FindCmdError
from traitlets.config import get_config
from traitlets.config.configurable import SingletonConfigurable
from traitlets import List, Bool, Unicode
from IPython.utils.py3compat import cast_unicode
def latex_to_png_mpl(s, wrap, color='Black', scale=1.0):
    try:
        from matplotlib import figure, font_manager, mathtext
        from matplotlib.backends import backend_agg
        from pyparsing import ParseFatalException
    except ImportError:
        return None
    s = s.replace('$$', '$')
    if wrap:
        s = u'${0}$'.format(s)
    try:
        prop = font_manager.FontProperties(size=12)
        dpi = 120 * scale
        buffer = BytesIO()
        parser = mathtext.MathTextParser('path')
        width, height, depth, _, _ = parser.parse(s, dpi=72, prop=prop)
        fig = figure.Figure(figsize=(width / 72, height / 72))
        fig.text(0, depth / height, s, fontproperties=prop, color=color)
        backend_agg.FigureCanvasAgg(fig)
        fig.savefig(buffer, dpi=dpi, format='png', transparent=True)
        return buffer.getvalue()
    except (ValueError, RuntimeError, ParseFatalException):
        return None