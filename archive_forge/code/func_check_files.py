from __future__ import annotations
import argparse
import logging
import os.path
from typing import Any
from flake8.discover_files import expand_paths
from flake8.formatting import base as formatter
from flake8.main import application as app
from flake8.options.parse_args import parse_args
def check_files(self, paths: list[str] | None=None) -> Report:
    """Run collected checks on the files provided.

        This will check the files passed in and return a :class:`Report`
        instance.

        :param paths:
            List of filenames (or paths) to check.
        :returns:
            Object that mimic's Flake8 2.0's Reporter class.
        """
    assert self._application.options is not None
    self._application.options.filenames = paths
    self._application.run_checks()
    self._application.report_errors()
    return Report(self._application)