import numpy as np
import pytest
import sympy
import cirq
def test_fsim_json_dict():
    assert cirq.FSimGate(theta=0.123, phi=0.456)._json_dict_() == {'theta': 0.123, 'phi': 0.456}