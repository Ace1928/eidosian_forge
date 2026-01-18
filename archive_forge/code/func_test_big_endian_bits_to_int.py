import pytest
import cirq
def test_big_endian_bits_to_int():
    assert cirq.big_endian_bits_to_int([0, 1]) == 1
    assert cirq.big_endian_bits_to_int([1, 0]) == 2
    assert cirq.big_endian_bits_to_int([0, 1, 0]) == 2
    assert cirq.big_endian_bits_to_int([1, 0, 0, 1, 0]) == 18
    assert cirq.big_endian_bits_to_int([]) == 0
    assert cirq.big_endian_bits_to_int([0]) == 0
    assert cirq.big_endian_bits_to_int([0, 0]) == 0
    assert cirq.big_endian_bits_to_int([0, 0, 0]) == 0
    assert cirq.big_endian_bits_to_int([1]) == 1
    assert cirq.big_endian_bits_to_int([0, 1]) == 1
    assert cirq.big_endian_bits_to_int([0, 0, 1]) == 1