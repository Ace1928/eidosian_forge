import dataclasses
import cirq
import cirq_google
import pytest
from cirq_google import (
def test_dict_round_trip():
    input_dict = dict(name='test', idx=5)
    kv = KeyValueExecutableSpec.from_dict(input_dict, executable_family='cirq_google.algo_benchmarks.example')
    actual_dict = kv.to_dict()
    assert input_dict == actual_dict