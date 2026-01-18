import re
import pytest
import sympy
import cirq
def test_key_condition_str():
    assert str(init_key_condition) == '0:a'
    assert str(cirq.KeyCondition(key_a, index=-2)) == '0:a[-2]'