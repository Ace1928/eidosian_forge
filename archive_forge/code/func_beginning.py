from __future__ import annotations
import argparse
import os
import sys
from typing import IO
from flake8.formatting import _windows_color
from flake8.statistics import Statistics
from flake8.violation import Violation
def beginning(self, filename: str) -> None:
    """Notify the formatter that we're starting to process a file.

        :param filename:
            The name of the file that Flake8 is beginning to report results
            from.
        """