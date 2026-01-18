import hashlib
from pathlib import Path
from docutils import nodes
from docutils.parsers.rst import Directive, directives
import sphinx
from sphinx.errors import ConfigError, ExtensionError
import matplotlib as mpl
from matplotlib import _api, mathtext
from matplotlib.rcsetup import validate_float_or_None
def latex2png(latex, filename, fontset='cm', fontsize=10, dpi=100):
    with mpl.rc_context({'mathtext.fontset': fontset, 'font.size': fontsize}):
        try:
            depth = mathtext.math_to_image(f'${latex}$', filename, dpi=dpi, format='png')
        except Exception:
            _api.warn_external(f'Could not render math expression {latex}')
            depth = 0
    return depth