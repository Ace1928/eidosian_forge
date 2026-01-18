import re
import pytest
import sympy
import cirq
def test_key_condition_with_keys():
    c = init_key_condition.replace_key(key_a, key_b)
    assert c.key is key_b
    c = init_key_condition.replace_key(key_b, key_c)
    assert c.key is key_a