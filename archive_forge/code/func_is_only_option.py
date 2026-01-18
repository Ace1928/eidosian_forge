import os
import sys
import shutil
import argparse
from textwrap import dedent
from pip._vendor.pygments import __version__, highlight
from pip._vendor.pygments.util import ClassNotFound, OptionError, docstring_headline, \
from pip._vendor.pygments.lexers import get_all_lexers, get_lexer_by_name, guess_lexer, \
from pip._vendor.pygments.lexers.special import TextLexer
from pip._vendor.pygments.formatters.latex import LatexEmbeddedLexer, LatexFormatter
from pip._vendor.pygments.formatters import get_all_formatters, get_formatter_by_name, \
from pip._vendor.pygments.formatters.terminal import TerminalFormatter
from pip._vendor.pygments.formatters.terminal256 import Terminal256Formatter, TerminalTrueColorFormatter
from pip._vendor.pygments.filters import get_all_filters, find_filter_class
from pip._vendor.pygments.styles import get_all_styles, get_style_by_name
def is_only_option(opt):
    return not any((v for k, v in vars(argns).items() if k != opt))