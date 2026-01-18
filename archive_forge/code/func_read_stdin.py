import sys
import argparse
from .renderers.rst import RSTRenderer
from .renderers.markdown import MarkdownRenderer
from . import (
def read_stdin():
    is_stdin_pipe = not sys.stdin.isatty()
    if is_stdin_pipe:
        return sys.stdin.read()
    else:
        return None