import cirq
def assert_equal_mod_empty(expected, actual):
    actual = cirq.drop_empty_moments(actual)
    cirq.testing.assert_same_circuits(actual, expected)