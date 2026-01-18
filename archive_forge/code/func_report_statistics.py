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
def report_statistics(self) -> None:
    """Aggregate and report statistics from this run."""
    assert self.options is not None
    if not self.options.statistics:
        return
    assert self.formatter is not None
    assert self.guide is not None
    self.formatter.show_statistics(self.guide.stats)