from __future__ import annotations
import argparse
import logging
import os.path
from typing import Any
from flake8.discover_files import expand_paths
from flake8.formatting import base as formatter
from flake8.main import application as app
from flake8.options.parse_args import parse_args
def get_statistics(self, violation: str) -> list[str]:
    """Get the list of occurrences of a violation.

        :returns:
            List of occurrences of a violation formatted as:
            {Count} {Error Code} {Message}, e.g.,
            ``8 E531 Some error message about the error``
        """
    return [f'{s.count} {s.error_code} {s.message}' for s in self._stats.statistics_for(violation)]