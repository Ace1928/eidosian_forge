import json
from typing import Callable, Tuple, cast
from adagio.exceptions import (DependencyDefinitionError,
from adagio.instances import _ConfigVar, _Input, _Output, _Task
from adagio.shells.interfaceless import function_to_taskspec
from adagio.specs import (ConfigSpec, InputSpec, OutputSpec, TaskSpec,
from pytest import raises
from triad.collections.dict import ParamDict
from triad.utils.hash import to_uuid
def test_inputspec():
    raises(AssertionError, lambda: InputSpec('', int, True, False, None))
    raises(TypeError, lambda: InputSpec('a', 'xyz', True, False, None))
    raises(AssertionError, lambda: InputSpec('a', 'int', False, False, None))
    raises(AssertionError, lambda: InputSpec('a', 'int', True, True, 1, default_on_timeout=False))
    raises(ValueError, lambda: InputSpec('a', 'int', True, False, 'abc'))
    assert 10 == InputSpec('a', 'int', True, False, '10').default_value
    InputSpec('a', int, False, True, None, default_on_timeout=False)
    InputSpec('a', int, True, True, None, default_on_timeout=False)
    s = InputSpec('a', _Task, True, False, None)
    raises(TypeError, lambda: s.validate_value(123))
    assert s.validate_value(None) is None
    t = MockTaskForVar()
    assert s.validate_value(t) is t
    s = InputSpec('a', _Task, True, True, None, default_on_timeout=False)
    assert s.validate_value(None) is None
    t = MockTaskForVar()
    assert s.validate_value(t) is t
    s = InputSpec('a', _Task, False, True, None, default_on_timeout=False)
    raises(AssertionError, lambda: s.validate_value(None))
    t = MockTaskForVar()
    assert s.validate_value(t) is t
    s = InputSpec('a', _Task, False, True, None, timeout=3, default_on_timeout=False)
    assert 3 == s.timeout
    assert not s.default_on_timeout
    o = OutputSpec('x', _Task, True)
    raises(TypeError, lambda: s.validate_spec(o))
    o = OutputSpec('x', _Task, False)
    assert o is s.validate_spec(o)
    s = InputSpec('a', _Task, True, True, None, timeout=3, default_on_timeout=False)
    o = OutputSpec('x', _Task, True)
    assert o is s.validate_spec(o)
    o = OutputSpec('x', _Task, False)
    assert o is s.validate_spec(o)