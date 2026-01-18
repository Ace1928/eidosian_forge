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
def perform_collect(self, args: Optional[Sequence[str]]=None, genitems: bool=True) -> Sequence[Union[nodes.Item, nodes.Collector]]:
    """Perform the collection phase for this session.

        This is called by the default :hook:`pytest_collection` hook
        implementation; see the documentation of this hook for more details.
        For testing purposes, it may also be called directly on a fresh
        ``Session``.

        This function normally recursively expands any collectors collected
        from the session to their items, and only items are returned. For
        testing purposes, this may be suppressed by passing ``genitems=False``,
        in which case the return value contains these collectors unexpanded,
        and ``session.items`` is empty.
        """
    if args is None:
        args = self.config.args
    self.trace('perform_collect', self, args)
    self.trace.root.indent += 1
    hook = self.config.hook
    self._notfound = []
    self._initial_parts = []
    self._collection_cache = {}
    self.items = []
    items: Sequence[Union[nodes.Item, nodes.Collector]] = self.items
    try:
        initialpaths: List[Path] = []
        initialpaths_with_parents: List[Path] = []
        for arg in args:
            collection_argument = resolve_collection_argument(self.config.invocation_params.dir, arg, as_pypath=self.config.option.pyargs)
            self._initial_parts.append(collection_argument)
            initialpaths.append(collection_argument.path)
            initialpaths_with_parents.append(collection_argument.path)
            initialpaths_with_parents.extend(collection_argument.path.parents)
        self._initialpaths = frozenset(initialpaths)
        self._initialpaths_with_parents = frozenset(initialpaths_with_parents)
        rep = collect_one_node(self)
        self.ihook.pytest_collectreport(report=rep)
        self.trace.root.indent -= 1
        if self._notfound:
            errors = []
            for arg, collectors in self._notfound:
                if collectors:
                    errors.append(f'not found: {arg}\n(no match in any of {collectors!r})')
                else:
                    errors.append(f'found no collectors for {arg}')
            raise UsageError(*errors)
        if not genitems:
            items = rep.result
        elif rep.passed:
            for node in rep.result:
                self.items.extend(self.genitems(node))
        self.config.pluginmanager.check_pending()
        hook.pytest_collection_modifyitems(session=self, config=self.config, items=items)
    finally:
        self._notfound = []
        self._initial_parts = []
        self._collection_cache = {}
        hook.pytest_collection_finish(session=self)
    if genitems:
        self.testscollected = len(items)
    return items