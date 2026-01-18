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
def runpytest_subprocess(self, *args: Union[str, 'os.PathLike[str]'], timeout: Optional[float]=None) -> RunResult:
    """Run pytest as a subprocess with given arguments.

        Any plugins added to the :py:attr:`plugins` list will be added using the
        ``-p`` command line option.  Additionally ``--basetemp`` is used to put
        any temporary files and directories in a numbered directory prefixed
        with "runpytest-" to not conflict with the normal numbered pytest
        location for temporary files and directories.

        :param args:
            The sequence of arguments to pass to the pytest subprocess.
        :param timeout:
            The period in seconds after which to timeout and raise
            :py:class:`Pytester.TimeoutExpired`.
        :returns:
            The result.
        """
    __tracebackhide__ = True
    p = make_numbered_dir(root=self.path, prefix='runpytest-', mode=448)
    args = ('--basetemp=%s' % p, *args)
    plugins = [x for x in self.plugins if isinstance(x, str)]
    if plugins:
        args = ('-p', plugins[0], *args)
    args = self._getpytestargs() + args
    return self.run(*args, timeout=timeout)