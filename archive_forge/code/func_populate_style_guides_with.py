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
def populate_style_guides_with(self, options: argparse.Namespace) -> Generator[StyleGuide, None, None]:
    """Generate style guides from the per-file-ignores option.

        :param options:
            The original options parsed from the CLI and config file.
        :returns:
            A copy of the default style guide with overridden values.
        """
    per_file = utils.parse_files_to_codes_mapping(options.per_file_ignores)
    for filename, violations in per_file:
        yield self.default_style_guide.copy(filename=filename, extend_ignore_with=violations)