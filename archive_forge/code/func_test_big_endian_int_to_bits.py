import pytest
import cirq
def test_big_endian_int_to_bits():
    assert cirq.big_endian_int_to_bits(2, bit_count=4) == [0, 0, 1, 0]
    assert cirq.big_endian_int_to_bits(18, bit_count=8) == [0, 0, 0, 1, 0, 0, 1, 0]
    assert cirq.big_endian_int_to_bits(18, bit_count=4) == [0, 0, 1, 0]
    assert cirq.big_endian_int_to_bits(-3, bit_count=4) == [1, 1, 0, 1]