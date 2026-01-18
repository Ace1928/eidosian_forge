import dataclasses
import inspect
import enum
import functools
import textwrap
from typing import (
from typing_extensions import Protocol
from cirq import circuits
@functools.wraps(func)
def func_with_logging(circuit: 'cirq.AbstractCircuit', **kwargs) -> 'cirq.AbstractCircuit':
    return _transform_and_log(add_deep_support, func, func.__name__, circuit, kwargs.get('context', default_context), **kwargs)