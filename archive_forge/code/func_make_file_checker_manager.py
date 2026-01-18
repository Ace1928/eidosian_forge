from __future__ import annotations
import argparse
import json
import logging
import time
from typing import Sequence
import flake8
from flake8 import checker
from flake8 import defaults
from flake8 import exceptions
from flake8 import style_guide
from flake8.formatting.base import BaseFormatter
from flake8.main import debug
from flake8.options.parse_args import parse_args
from flake8.plugins import finder
from flake8.plugins import reporter
def make_file_checker_manager(self, argv: Sequence[str]) -> None:
    """Initialize our FileChecker Manager."""
    assert self.guide is not None
    assert self.plugins is not None
    self.file_checker_manager = checker.Manager(style_guide=self.guide, plugins=self.plugins.checkers, argv=argv)