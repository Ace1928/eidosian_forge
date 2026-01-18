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
def check_physical_eol(self, token: tokenize.TokenInfo, prev_physical: str) -> None:
    """Run physical checks if and only if it is at the end of the line."""
    assert self.processor is not None
    if token.type == FSTRING_START:
        self.processor.fstring_start(token.start[0])
    elif processor.is_eol_token(token):
        if token.line == '':
            self.run_physical_checks(prev_physical)
        else:
            self.run_physical_checks(token.line)
    elif processor.is_multiline_string(token):
        for line in self.processor.multiline_string(token):
            self.run_physical_checks(line)