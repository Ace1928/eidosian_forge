import collections
import importlib.machinery
import io
import linecache
import pickletools
import platform
import types
from collections import defaultdict, OrderedDict
from dataclasses import dataclass
from enum import Enum
from importlib.machinery import SourceFileLoader
from pathlib import Path
from typing import (
import torch
from torch.serialization import location_tag, normalize_storage_type
from torch.types import Storage
from torch.utils.hooks import RemovableHandle
from ._digraph import DiGraph
from ._importlib import _normalize_path
from ._mangling import demangle, is_mangled
from ._package_pickler import create_pickler
from ._stdlib import is_stdlib_module
from .find_file_dependencies import find_files_source_depends_on
from .glob_group import GlobGroup, GlobPattern
from .importer import Importer, OrderedImporter, sys_importer
from _mock import MockedObject
def save_pickle(self, package: str, resource: str, obj: Any, dependencies: bool=True, pickle_protocol: int=3):
    """Save a python object to the archive using pickle. Equivalent to :func:`torch.save` but saving into
        the archive rather than a stand-alone file. Standard pickle does not save the code, only the objects.
        If ``dependencies`` is true, this method will also scan the pickled objects for which modules are required
        to reconstruct them and save the relevant code.

        To be able to save an object where ``type(obj).__name__`` is ``my_module.MyObject``,
        ``my_module.MyObject`` must resolve to the class of the object according to the ``importer`` order. When saving objects that
        have previously been packaged, the importer's ``import_module`` method will need to be present in the ``importer`` list
        for this to work.

        Args:
            package (str): The name of module package this resource should go in (e.g. ``"my_package.my_subpackage"``).
            resource (str): A unique name for the resource, used to identify it to load.
            obj (Any): The object to save, must be picklable.
            dependencies (bool, optional): If ``True``, we scan the source for dependencies.
        """
    assert pickle_protocol == 4 or pickle_protocol == 3, 'torch.package only supports pickle protocols 3 and 4'
    filename = self._filename(package, resource)
    data_buf = io.BytesIO()
    pickler = create_pickler(data_buf, self.importer, protocol=pickle_protocol)
    pickler.persistent_id = self._persistent_id
    pickler.dump(obj)
    data_value = data_buf.getvalue()
    mocked_modules = defaultdict(list)
    name_in_dependency_graph = f'<{package}.{resource}>'
    self.dependency_graph.add_node(name_in_dependency_graph, action=_ModuleProviderAction.INTERN, provided=True, is_pickle=True)

    def _check_mocked_error(module: Optional[str], field: Optional[str]):
        """
            checks if an object (field) comes from a mocked module and then adds
            the pair to mocked_modules which contains mocked modules paired with their
            list of mocked objects present in the pickle.

            We also hold the invariant that the first user defined rule that applies
            to the module is the one we use.
            """
        assert isinstance(module, str)
        assert isinstance(field, str)
        if self._can_implicitly_extern(module):
            return
        for pattern, pattern_info in self.patterns.items():
            if pattern.matches(module):
                if pattern_info.action == _ModuleProviderAction.MOCK:
                    mocked_modules[module].append(field)
                return
    if dependencies:
        all_dependencies = []
        module = None
        field = None
        memo: DefaultDict[int, str] = defaultdict(None)
        memo_count = 0
        for opcode, arg, pos in pickletools.genops(data_value):
            if pickle_protocol == 4:
                if opcode.name == 'SHORT_BINUNICODE' or opcode.name == 'BINUNICODE' or opcode.name == 'BINUNICODE8':
                    assert isinstance(arg, str)
                    module = field
                    field = arg
                    memo[memo_count] = arg
                elif opcode.name == 'LONG_BINGET' or opcode.name == 'BINGET' or opcode.name == 'GET':
                    assert isinstance(arg, int)
                    module = field
                    field = memo.get(arg, None)
                elif opcode.name == 'MEMOIZE':
                    memo_count += 1
                elif opcode.name == 'STACK_GLOBAL':
                    if module is None:
                        continue
                    assert isinstance(module, str)
                    if module not in all_dependencies:
                        all_dependencies.append(module)
                    _check_mocked_error(module, field)
            elif pickle_protocol == 3 and opcode.name == 'GLOBAL':
                assert isinstance(arg, str)
                module, field = arg.split(' ')
                if module not in all_dependencies:
                    all_dependencies.append(module)
                _check_mocked_error(module, field)
        for module_name in all_dependencies:
            self.dependency_graph.add_edge(name_in_dependency_graph, module_name)
            ' If an object happens to come from a mocked module, then we collect these errors and spit them\n                    out with the other errors found by package exporter.\n                '
            if module in mocked_modules:
                assert isinstance(module, str)
                fields = mocked_modules[module]
                self.dependency_graph.add_node(module_name, action=_ModuleProviderAction.MOCK, error=PackagingErrorReason.MOCKED_BUT_STILL_USED, error_context=f"Object(s) '{fields}' from module `{module_name}` was mocked out during packaging but is being used in resource - `{resource}` in package `{package}`. ", provided=True)
            else:
                self.add_dependency(module_name)
    self._write(filename, data_value)