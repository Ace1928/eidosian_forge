import json
from typing import Callable, cast
from adagio.exceptions import SkippedError
from adagio.instances import (_ConfigVar, _Dependency, _DependencyDict, _Input,
from adagio.specs import ConfigSpec, InputSpec, OutputSpec, TaskSpec
from pytest import raises
from triad.collections.dict import IndexedOrderedDict, ParamDict
from triad.exceptions import InvalidOperationError
from triad.utils.hash import to_uuid
def test_dependency():
    a = _Dependency()
    b = _Dependency().set_dependency(a)
    c = _Dependency().set_dependency(b)
    d = _Dependency().set_dependency(c)
    assert d.dependency is a