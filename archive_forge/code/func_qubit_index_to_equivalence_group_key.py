import collections.abc
import pathlib
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def qubit_index_to_equivalence_group_key(self, index: int) -> int:
    if index % 2 == 0:
        return index
    return 0