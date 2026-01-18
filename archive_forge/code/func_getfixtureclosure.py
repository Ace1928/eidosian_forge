import abc
from collections import defaultdict
from collections import deque
import dataclasses
import functools
import inspect
import os
from pathlib import Path
import sys
from typing import AbstractSet
from typing import Any
from typing import Callable
from typing import cast
from typing import Dict
from typing import Final
from typing import final
from typing import Generator
from typing import Generic
from typing import Iterable
from typing import Iterator
from typing import List
from typing import MutableMapping
from typing import NoReturn
from typing import Optional
from typing import overload
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import TYPE_CHECKING
from typing import TypeVar
from typing import Union
import warnings
import _pytest
from _pytest import nodes
from _pytest._code import getfslineno
from _pytest._code.code import FormattedExcinfo
from _pytest._code.code import TerminalRepr
from _pytest._io import TerminalWriter
from _pytest.compat import _PytestWrapper
from _pytest.compat import assert_never
from _pytest.compat import get_real_func
from _pytest.compat import get_real_method
from _pytest.compat import getfuncargnames
from _pytest.compat import getimfunc
from _pytest.compat import getlocation
from _pytest.compat import is_generator
from _pytest.compat import NOTSET
from _pytest.compat import NotSetType
from _pytest.compat import safe_getattr
from _pytest.config import _PluggyPlugin
from _pytest.config import Config
from _pytest.config.argparsing import Parser
from _pytest.deprecated import check_ispytest
from _pytest.deprecated import MARKED_FIXTURE
from _pytest.deprecated import YIELD_FIXTURE
from _pytest.mark import Mark
from _pytest.mark import ParameterSet
from _pytest.mark.structures import MarkDecorator
from _pytest.outcomes import fail
from _pytest.outcomes import skip
from _pytest.outcomes import TEST_OUTCOME
from _pytest.pathlib import absolutepath
from _pytest.pathlib import bestrelpath
from _pytest.scope import _ScopeName
from _pytest.scope import HIGH_SCOPES
from _pytest.scope import Scope
def getfixtureclosure(self, parentnode: nodes.Node, initialnames: Tuple[str, ...], ignore_args: AbstractSet[str]) -> Tuple[List[str], Dict[str, Sequence[FixtureDef[Any]]]]:
    fixturenames_closure = list(initialnames)
    arg2fixturedefs: Dict[str, Sequence[FixtureDef[Any]]] = {}
    lastlen = -1
    while lastlen != len(fixturenames_closure):
        lastlen = len(fixturenames_closure)
        for argname in fixturenames_closure:
            if argname in ignore_args:
                continue
            if argname in arg2fixturedefs:
                continue
            fixturedefs = self.getfixturedefs(argname, parentnode)
            if fixturedefs:
                arg2fixturedefs[argname] = fixturedefs
                for arg in fixturedefs[-1].argnames:
                    if arg not in fixturenames_closure:
                        fixturenames_closure.append(arg)

    def sort_by_scope(arg_name: str) -> Scope:
        try:
            fixturedefs = arg2fixturedefs[arg_name]
        except KeyError:
            return Scope.Function
        else:
            return fixturedefs[-1]._scope
    fixturenames_closure.sort(key=sort_by_scope, reverse=True)
    return (fixturenames_closure, arg2fixturedefs)