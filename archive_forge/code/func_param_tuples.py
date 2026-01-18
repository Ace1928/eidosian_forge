from typing import (
import abc
import collections
import itertools
import sympy
from cirq import protocols
from cirq._doc import document
from cirq.study import resolver
def param_tuples(self) -> Iterator[Params]:
    for r in self.resolver_list:
        yield tuple(_params_without_symbols(r))