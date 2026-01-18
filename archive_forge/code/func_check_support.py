import math
import numbers
from typing import cast, Dict, FrozenSet, Iterable, Iterator, List, Optional, Sequence, Union
import numpy as np
import sympy
from cirq_google.api import v2
from cirq_google.ops import InternalGate
def check_support(func_type: str) -> str:
    if func_type not in supported:
        lang = repr(arg_function_language) if arg_function_language is not None else '[any]'
        raise ValueError(f'Function type {func_type!r} not supported by arg_function_language {lang}')
    return func_type