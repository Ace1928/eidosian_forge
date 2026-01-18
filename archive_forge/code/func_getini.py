import argparse
import collections.abc
import copy
import dataclasses
import enum
from functools import lru_cache
import glob
import importlib.metadata
import inspect
import os
from pathlib import Path
import re
import shlex
import sys
from textwrap import dedent
import types
from types import FunctionType
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import Final
from typing import final
from typing import Generator
from typing import IO
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Optional
from typing import Sequence
from typing import Set
from typing import TextIO
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import Union
import warnings
import pluggy
from pluggy import HookimplMarker
from pluggy import HookimplOpts
from pluggy import HookspecMarker
from pluggy import HookspecOpts
from pluggy import PluginManager
from .compat import PathAwareHookProxy
from .exceptions import PrintHelp as PrintHelp
from .exceptions import UsageError as UsageError
from .findpaths import determine_setup
import _pytest._code
from _pytest._code import ExceptionInfo
from _pytest._code import filter_traceback
from _pytest._io import TerminalWriter
import _pytest.deprecated
import _pytest.hookspec
from _pytest.outcomes import fail
from _pytest.outcomes import Skipped
from _pytest.pathlib import absolutepath
from _pytest.pathlib import bestrelpath
from _pytest.pathlib import import_path
from _pytest.pathlib import ImportMode
from _pytest.pathlib import resolve_package_path
from _pytest.pathlib import safe_exists
from _pytest.stash import Stash
from _pytest.warning_types import PytestConfigWarning
from _pytest.warning_types import warn_explicit_for
def getini(self, name: str):
    """Return configuration value from an :ref:`ini file <configfiles>`.

        If a configuration value is not defined in an
        :ref:`ini file <configfiles>`, then the ``default`` value provided while
        registering the configuration through
        :func:`parser.addini <pytest.Parser.addini>` will be returned.
        Please note that you can even provide ``None`` as a valid
        default value.

        If ``default`` is not provided while registering using
        :func:`parser.addini <pytest.Parser.addini>`, then a default value
        based on the ``type`` parameter passed to
        :func:`parser.addini <pytest.Parser.addini>` will be returned.
        The default values based on ``type`` are:
        ``paths``, ``pathlist``, ``args`` and ``linelist`` : empty list ``[]``
        ``bool`` : ``False``
        ``string`` : empty string ``""``

        If neither the ``default`` nor the ``type`` parameter is passed
        while registering the configuration through
        :func:`parser.addini <pytest.Parser.addini>`, then the configuration
        is treated as a string and a default empty string '' is returned.

        If the specified name hasn't been registered through a prior
        :func:`parser.addini <pytest.Parser.addini>` call (usually from a
        plugin), a ValueError is raised.
        """
    try:
        return self._inicache[name]
    except KeyError:
        self._inicache[name] = val = self._getini(name)
        return val