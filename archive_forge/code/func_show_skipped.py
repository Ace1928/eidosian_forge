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
def show_skipped(lines: List[str]) -> None:
    skipped: List[CollectReport] = self.stats.get('skipped', [])
    fskips = _folded_skips(self.startpath, skipped) if skipped else []
    if not fskips:
        return
    verbose_word = skipped[0]._get_verbose_word(self.config)
    markup_word = self._tw.markup(verbose_word, **{_color_for_type['warnings']: True})
    prefix = 'Skipped: '
    for num, fspath, lineno, reason in fskips:
        if reason.startswith(prefix):
            reason = reason[len(prefix):]
        if lineno is not None:
            lines.append('%s [%d] %s:%d: %s' % (markup_word, num, fspath, lineno, reason))
        else:
            lines.append('%s [%d] %s: %s' % (markup_word, num, fspath, reason))