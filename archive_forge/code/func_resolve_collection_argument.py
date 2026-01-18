import argparse
import dataclasses
import fnmatch
import functools
import importlib
import importlib.util
import os
from pathlib import Path
import sys
from typing import AbstractSet
from typing import Callable
from typing import Dict
from typing import final
from typing import FrozenSet
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Literal
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Tuple
from typing import TYPE_CHECKING
from typing import Union
import warnings
import pluggy
from _pytest import nodes
import _pytest._code
from _pytest.config import Config
from _pytest.config import directory_arg
from _pytest.config import ExitCode
from _pytest.config import hookimpl
from _pytest.config import PytestPluginManager
from _pytest.config import UsageError
from _pytest.config.argparsing import Parser
from _pytest.config.compat import PathAwareHookProxy
from _pytest.fixtures import FixtureManager
from _pytest.outcomes import exit
from _pytest.pathlib import absolutepath
from _pytest.pathlib import bestrelpath
from _pytest.pathlib import fnmatch_ex
from _pytest.pathlib import safe_exists
from _pytest.pathlib import scandir
from _pytest.reports import CollectReport
from _pytest.reports import TestReport
from _pytest.runner import collect_one_node
from _pytest.runner import SetupState
from _pytest.warning_types import PytestWarning
def resolve_collection_argument(invocation_path: Path, arg: str, *, as_pypath: bool=False) -> CollectionArgument:
    """Parse path arguments optionally containing selection parts and return (fspath, names).

    Command-line arguments can point to files and/or directories, and optionally contain
    parts for specific tests selection, for example:

        "pkg/tests/test_foo.py::TestClass::test_foo"

    This function ensures the path exists, and returns a resolved `CollectionArgument`:

        CollectionArgument(
            path=Path("/full/path/to/pkg/tests/test_foo.py"),
            parts=["TestClass", "test_foo"],
            module_name=None,
        )

    When as_pypath is True, expects that the command-line argument actually contains
    module paths instead of file-system paths:

        "pkg.tests.test_foo::TestClass::test_foo"

    In which case we search sys.path for a matching module, and then return the *path* to the
    found module, which may look like this:

        CollectionArgument(
            path=Path("/home/u/myvenv/lib/site-packages/pkg/tests/test_foo.py"),
            parts=["TestClass", "test_foo"],
            module_name="pkg.tests.test_foo",
        )

    If the path doesn't exist, raise UsageError.
    If the path is a directory and selection parts are present, raise UsageError.
    """
    base, squacket, rest = str(arg).partition('[')
    strpath, *parts = base.split('::')
    if parts:
        parts[-1] = f'{parts[-1]}{squacket}{rest}'
    module_name = None
    if as_pypath:
        pyarg_strpath = search_pypath(strpath)
        if pyarg_strpath is not None:
            module_name = strpath
            strpath = pyarg_strpath
    fspath = invocation_path / strpath
    fspath = absolutepath(fspath)
    if not safe_exists(fspath):
        msg = 'module or package not found: {arg} (missing __init__.py?)' if as_pypath else 'file or directory not found: {arg}'
        raise UsageError(msg.format(arg=arg))
    if parts and fspath.is_dir():
        msg = 'package argument cannot contain :: selection parts: {arg}' if as_pypath else 'directory argument cannot contain :: selection parts: {arg}'
        raise UsageError(msg.format(arg=arg))
    return CollectionArgument(path=fspath, parts=parts, module_name=module_name)