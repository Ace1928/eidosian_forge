import pytest
import numpy as np
import cirq
def test_assert_specifies_has_unitary_if_unitary_pass():

    class Good:

        def _has_unitary_(self):
            return True
    assert cirq.has_unitary(Good())
    cirq.testing.assert_specifies_has_unitary_if_unitary(Good())