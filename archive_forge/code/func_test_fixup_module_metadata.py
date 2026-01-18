import signal
import sys
import types
from typing import Any, TypeVar
import pytest
import trio
from trio.testing import Matcher, RaisesGroup
from .. import _core
from .._core._tests.tutil import (
from .._util import (
from ..testing import wait_all_tasks_blocked
def test_fixup_module_metadata() -> None:
    non_trio_module = types.ModuleType('not_trio')
    non_trio_module.some_func = lambda: None
    non_trio_module.some_func.__name__ = 'some_func'
    non_trio_module.some_func.__qualname__ = 'some_func'
    fixup_module_metadata(non_trio_module.__name__, vars(non_trio_module))
    assert non_trio_module.some_func.__name__ == 'some_func'
    assert non_trio_module.some_func.__qualname__ == 'some_func'
    mod = types.ModuleType('trio._somemodule_impl')
    mod.some_func = lambda: None
    mod.some_func.__name__ = '_something_else'
    mod.some_func.__qualname__ = '_something_else'
    mod.not_funclike = types.SimpleNamespace()
    mod.not_funclike.__name__ = 'not_funclike'
    mod.only_has_name = types.SimpleNamespace()
    mod.only_has_name.__module__ = 'trio._somemodule_impl'
    mod.only_has_name.__name__ = 'only_name'
    mod._private = lambda: None
    mod._private.__module__ = 'trio._somemodule_impl'
    mod._private.__name__ = mod._private.__qualname__ = '_private'
    mod.SomeClass = type('SomeClass', (), {'__init__': lambda self: None, 'method': lambda self: None})
    mod.SomeClass.recursion = mod.SomeClass
    fixup_module_metadata('trio.somemodule', vars(mod))
    assert mod.some_func.__name__ == 'some_func'
    assert mod.some_func.__module__ == 'trio.somemodule'
    assert mod.some_func.__qualname__ == 'some_func'
    assert mod.not_funclike.__name__ == 'not_funclike'
    assert mod._private.__name__ == '_private'
    assert mod._private.__module__ == 'trio._somemodule_impl'
    assert mod._private.__qualname__ == '_private'
    assert mod.only_has_name.__name__ == 'only_has_name'
    assert mod.only_has_name.__module__ == 'trio.somemodule'
    assert not hasattr(mod.only_has_name, '__qualname__')
    assert mod.SomeClass.method.__name__ == 'method'
    assert mod.SomeClass.method.__module__ == 'trio.somemodule'
    assert mod.SomeClass.method.__qualname__ == 'SomeClass.method'
    non_trio_module.some_func()
    mod.some_func()
    mod._private()
    mod.SomeClass().method()