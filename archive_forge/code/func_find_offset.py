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
def find_offset(offset: int, mapping: processor._LogicalMapping) -> tuple[int, int]:
    """Find the offset tuple for a single offset."""
    if isinstance(offset, tuple):
        return offset
    for token in mapping:
        token_offset = token[0]
        if offset <= token_offset:
            position = token[1]
            break
    else:
        position = (0, 0)
        offset = token_offset = 0
    return (position[0], position[1] + offset - token_offset)