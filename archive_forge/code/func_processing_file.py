from __future__ import annotations
import argparse
import contextlib
import copy
import enum
import functools
import logging
from typing import Generator
from typing import Sequence
from flake8 import defaults
from flake8 import statistics
from flake8 import utils
from flake8.formatting import base as base_formatter
from flake8.violation import Violation
@contextlib.contextmanager
def processing_file(self, filename: str) -> Generator[StyleGuide, None, None]:
    """Record the fact that we're processing the file's results."""
    self.formatter.beginning(filename)
    yield self
    self.formatter.finished(filename)