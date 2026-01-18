from typing import Dict, Optional, Sequence
import numpy as np
import cirq
from cirq import circuits
def assert_equivalent_computational_basis_map(maps: Dict[int, int], circuit: circuits.Circuit):
    """Ensure equivalence of basis state mapping.

    Args:
        maps: dictionary of test computational basis input states and
            the output computational basis states that they should be mapped to.
            The states are specified using little endian convention, meaning
            that the last bit of a binary literal will refer to the last qubit's
            value.
        circuit: the circuit to be tested
    Raises:
        AssertionError: Raised in case any basis state is mapped to the wrong
            basis state.
    """
    keys = sorted(maps.keys())
    actual_map = _sparse_computational_basis_map(keys, circuit)
    mbl = max(keys).bit_length()
    for k in keys:
        assert actual_map.get(k) == maps[k], f'{_bin_dec(k, mbl)} was mapped to {_bin_dec(actual_map.get(k), mbl)} instead of {_bin_dec(maps[k], mbl)}.'