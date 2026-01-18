import bdb
import builtins
import inspect
import os
import platform
import sys
import traceback
import types
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import (
import pytest
from _pytest import outcomes
from _pytest._code.code import ExceptionInfo, ReprFileLocation, TerminalRepr
from _pytest._io import TerminalWriter
from _pytest.compat import safe_getattr
from _pytest.config import Config
from _pytest.config.argparsing import Parser
from _pytest.fixtures import FixtureRequest
from _pytest.nodes import Collector
from _pytest.outcomes import OutcomeException
from _pytest.pathlib import fnmatch_ex, import_path
from _pytest.python_api import approx
from _pytest.warning_types import PytestWarning
def get_optionflags(parent):
    optionflags_str = parent.config.getini('ipdoctest_optionflags')
    flag_lookup_table = _get_flag_lookup()
    flag_acc = 0
    for flag in optionflags_str:
        flag_acc |= flag_lookup_table[flag]
    return flag_acc