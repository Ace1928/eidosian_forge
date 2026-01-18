import collections.abc
import contextlib
from fnmatch import fnmatch
import gc
import importlib
from io import StringIO
import locale
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import traceback
from typing import Any
from typing import Callable
from typing import Dict
from typing import Final
from typing import final
from typing import Generator
from typing import IO
from typing import Iterable
from typing import List
from typing import Literal
from typing import Optional
from typing import overload
from typing import Sequence
from typing import TextIO
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import Union
from weakref import WeakKeyDictionary
from iniconfig import IniConfig
from iniconfig import SectionWrapper
from _pytest import timing
from _pytest._code import Source
from _pytest.capture import _get_multicapture
from _pytest.compat import NOTSET
from _pytest.compat import NotSetType
from _pytest.config import _PluggyPlugin
from _pytest.config import Config
from _pytest.config import ExitCode
from _pytest.config import hookimpl
from _pytest.config import main
from _pytest.config import PytestPluginManager
from _pytest.config.argparsing import Parser
from _pytest.deprecated import check_ispytest
from _pytest.fixtures import fixture
from _pytest.fixtures import FixtureRequest
from _pytest.main import Session
from _pytest.monkeypatch import MonkeyPatch
from _pytest.nodes import Collector
from _pytest.nodes import Item
from _pytest.outcomes import fail
from _pytest.outcomes import importorskip
from _pytest.outcomes import skip
from _pytest.pathlib import bestrelpath
from _pytest.pathlib import make_numbered_dir
from _pytest.reports import CollectReport
from _pytest.reports import TestReport
from _pytest.tmpdir import TempPathFactory
from _pytest.warning_types import PytestWarning
def matchreport(self, inamepart: str='', names: Union[str, Iterable[str]]=('pytest_runtest_logreport', 'pytest_collectreport'), when: Optional[str]=None) -> Union[CollectReport, TestReport]:
    """Return a testreport whose dotted import path matches."""
    values = []
    for rep in self.getreports(names=names):
        if not when and rep.when != 'call' and rep.passed:
            continue
        if when and rep.when != when:
            continue
        if not inamepart or inamepart in rep.nodeid.split('::'):
            values.append(rep)
    if not values:
        raise ValueError(f'could not find test report matching {inamepart!r}: no test reports at all!')
    if len(values) > 1:
        raise ValueError(f'found 2 or more testreports matching {inamepart!r}: {values}')
    return values[0]