from typing import Any, List
import abc
import sympy
import cirq
from cirq_google.api import v2
from cirq_google.serialization import arg_func_langs
@property
def serialized_id(self):
    return 'circuit'