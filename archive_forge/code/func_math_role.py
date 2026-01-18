import hashlib
from pathlib import Path
from docutils import nodes
from docutils.parsers.rst import Directive, directives
import sphinx
from sphinx.errors import ConfigError, ExtensionError
import matplotlib as mpl
from matplotlib import _api, mathtext
from matplotlib.rcsetup import validate_float_or_None
def math_role(role, rawtext, text, lineno, inliner, options={}, content=[]):
    i = rawtext.find('`')
    latex = rawtext[i + 1:-1]
    node = latex_math(rawtext)
    node['latex'] = latex
    node['fontset'] = options.get('fontset', 'cm')
    node['fontsize'] = options.get('fontsize', setup.app.config.mathmpl_fontsize)
    return ([node], [])