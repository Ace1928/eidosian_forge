import argparse
from collections import Counter
import dataclasses
import datetime
from functools import partial
import inspect
from pathlib import Path
import platform
import sys
import textwrap
from typing import Any
from typing import Callable
from typing import ClassVar
from typing import Dict
from typing import final
from typing import Generator
from typing import List
from typing import Literal
from typing import Mapping
from typing import NamedTuple
from typing import Optional
from typing import Sequence
from typing import Set
from typing import TextIO
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union
import warnings
import pluggy
from _pytest import nodes
from _pytest import timing
from _pytest._code import ExceptionInfo
from _pytest._code.code import ExceptionRepr
from _pytest._io import TerminalWriter
from _pytest._io.wcwidth import wcswidth
import _pytest._version
from _pytest.assertion.util import running_on_ci
from _pytest.config import _PluggyPlugin
from _pytest.config import Config
from _pytest.config import ExitCode
from _pytest.config import hookimpl
from _pytest.config.argparsing import Parser
from _pytest.nodes import Item
from _pytest.nodes import Node
from _pytest.pathlib import absolutepath
from _pytest.pathlib import bestrelpath
from _pytest.reports import BaseReport
from _pytest.reports import CollectReport
from _pytest.reports import TestReport
def build_summary_stats_line(self) -> Tuple[List[Tuple[str, Dict[str, bool]]], str]:
    """
        Build the parts used in the last summary stats line.

        The summary stats line is the line shown at the end, "=== 12 passed, 2 errors in Xs===".

        This function builds a list of the "parts" that make up for the text in that line, in
        the example above it would be:

            [
                ("12 passed", {"green": True}),
                ("2 errors", {"red": True}
            ]

        That last dict for each line is a "markup dictionary", used by TerminalWriter to
        color output.

        The final color of the line is also determined by this function, and is the second
        element of the returned tuple.
        """
    if self.config.getoption('collectonly'):
        return self._build_collect_only_summary_stats_line()
    else:
        return self._build_normal_summary_stats_line()