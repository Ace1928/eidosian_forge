import os
import re
import sys
from getopt import getopt, GetoptError
from traitlets.config.configurable import Configurable
from . import oinspect
from .error import UsageError
from .inputtransformer2 import ESC_MAGIC, ESC_MAGIC2
from ..utils.ipstruct import Struct
from ..utils.process import arg_split
from ..utils.text import dedent
from traitlets import Bool, Dict, Instance, observe
from logging import error
import typing as t
def format_latex(self, strng):
    """Format a string for latex inclusion."""
    escape_re = re.compile('(%|_|\\$|#|&)', re.MULTILINE)
    cmd_name_re = re.compile('^(%s.*?):' % ESC_MAGIC, re.MULTILINE)
    cmd_re = re.compile('(?P<cmd>%s.+?\\b)(?!\\}\\}:)' % ESC_MAGIC, re.MULTILINE)
    par_re = re.compile('\\\\$', re.MULTILINE)
    newline_re = re.compile('\\\\n')
    strng = cmd_name_re.sub('\\n\\\\bigskip\\n\\\\texttt{\\\\textbf{ \\1}}:', strng)
    strng = cmd_re.sub('\\\\texttt{\\g<cmd>}', strng)
    strng = par_re.sub('\\\\\\\\', strng)
    strng = escape_re.sub('\\\\\\1', strng)
    strng = newline_re.sub('\\\\textbackslash{}n', strng)
    return strng