from __future__ import annotations
import argparse
import logging
import os.path
from typing import Any
from flake8.discover_files import expand_paths
from flake8.formatting import base as formatter
from flake8.main import application as app
from flake8.options.parse_args import parse_args
def init_report(self, reporter: type[formatter.BaseFormatter] | None=None) -> None:
    """Set up a formatter for this run of Flake8."""
    if reporter is None:
        return
    if not issubclass(reporter, formatter.BaseFormatter):
        raise ValueError('Report should be subclass of flake8.formatter.BaseFormatter.')
    self._application.formatter = reporter(self.options)
    self._application.guide = None
    self._application.make_guide()
    self._application.file_checker_manager = None
    self._application.make_file_checker_manager([])