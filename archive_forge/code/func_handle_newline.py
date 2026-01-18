from __future__ import annotations
import argparse
import contextlib
import errno
import logging
import multiprocessing.pool
import operator
import signal
import tokenize
from typing import Any
from typing import Generator
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from flake8 import defaults
from flake8 import exceptions
from flake8 import processor
from flake8 import utils
from flake8._compat import FSTRING_START
from flake8.discover_files import expand_paths
from flake8.options.parse_args import parse_args
from flake8.plugins.finder import Checkers
from flake8.plugins.finder import LoadedPlugin
from flake8.style_guide import StyleGuideManager
def handle_newline(self, token_type: int) -> None:
    """Handle the logic when encountering a newline token."""
    assert self.processor is not None
    if token_type == tokenize.NEWLINE:
        self.run_logical_checks()
        self.processor.reset_blank_before()
    elif len(self.processor.tokens) == 1:
        self.processor.visited_new_blank_line()
        self.processor.delete_first_token()
    else:
        self.run_logical_checks()