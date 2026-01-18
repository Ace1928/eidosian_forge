from __future__ import annotations
import argparse
import logging
import os.path
from typing import Any
from flake8.discover_files import expand_paths
from flake8.formatting import base as formatter
from flake8.main import application as app
from flake8.options.parse_args import parse_args
def input_file(self, filename: str, lines: Any | None=None, expected: Any | None=None, line_offset: Any | None=0) -> Report:
    """Run collected checks on a single file.

        This will check the file passed in and return a :class:`Report`
        instance.

        :param filename:
            The path to the file to check.
        :param lines:
            Ignored since Flake8 3.0.
        :param expected:
            Ignored since Flake8 3.0.
        :param line_offset:
            Ignored since Flake8 3.0.
        :returns:
            Object that mimic's Flake8 2.0's Reporter class.
        """
    return self.check_files([filename])