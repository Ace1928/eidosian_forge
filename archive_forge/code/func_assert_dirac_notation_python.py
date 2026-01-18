import numpy as np
import pytest
import cirq
import cirq.testing
def assert_dirac_notation_python(vec, expected, decimals=2):
    assert cirq.dirac_notation(vec, decimals=decimals) == expected