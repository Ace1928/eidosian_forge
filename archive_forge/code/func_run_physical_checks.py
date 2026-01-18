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
def run_physical_checks(self, physical_line: str) -> None:
    """Run all checks for a given physical line.

        A single physical check may return multiple errors.
        """
    assert self.processor is not None
    for plugin in self.plugins.physical_line:
        self.processor.update_checker_state_for(plugin)
        result = self.run_check(plugin, physical_line=physical_line)
        if result is not None:
            column_offset = None
            try:
                column_offset = result[0]
            except (IndexError, TypeError):
                pass
            if isinstance(column_offset, int):
                result = (result,)
            for result_single in result:
                column_offset, text = result_single
                self.report(error_code=None, line_number=self.processor.line_number, column=column_offset, text=text)