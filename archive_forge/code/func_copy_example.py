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
def copy_example(self, name: Optional[str]=None) -> Path:
    """Copy file from project's directory into the testdir.

        :param name:
            The name of the file to copy.
        :return:
            Path to the copied directory (inside ``self.path``).
        """
    example_dir_ = self._request.config.getini('pytester_example_dir')
    if example_dir_ is None:
        raise ValueError("pytester_example_dir is unset, can't copy examples")
    example_dir: Path = self._request.config.rootpath / example_dir_
    for extra_element in self._request.node.iter_markers('pytester_example_path'):
        assert extra_element.args
        example_dir = example_dir.joinpath(*extra_element.args)
    if name is None:
        func_name = self._name
        maybe_dir = example_dir / func_name
        maybe_file = example_dir / (func_name + '.py')
        if maybe_dir.is_dir():
            example_path = maybe_dir
        elif maybe_file.is_file():
            example_path = maybe_file
        else:
            raise LookupError(f"{func_name} can't be found as module or package in {example_dir}")
    else:
        example_path = example_dir.joinpath(name)
    if example_path.is_dir() and (not example_path.joinpath('__init__.py').is_file()):
        shutil.copytree(example_path, self.path, symlinks=True, dirs_exist_ok=True)
        return self.path
    elif example_path.is_file():
        result = self.path.joinpath(example_path.name)
        shutil.copy(example_path, result)
        return result
    else:
        raise LookupError(f'example "{example_path}" is not found as a file or directory')