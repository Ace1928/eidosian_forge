import collections
import numpy as np
import pandas as pd
import pytest
import cirq
import cirq.testing
from cirq.study.result import _pack_digits
@pytest.mark.parametrize('use_records', [False, True])
def test_json_bit_packing_and_dtype(use_records: bool) -> None:
    shape = (256, 3, 256) if use_records else (256, 256)
    prng = np.random.RandomState(1234)
    bits = prng.randint(2, size=shape).astype(np.uint8)
    digits = prng.randint(256, size=shape).astype(np.uint8)
    params = cirq.ParamResolver({})
    if use_records:
        bits_result = cirq.ResultDict(params=params, records={'m': bits})
        digits_result = cirq.ResultDict(params=params, records={'m': digits})
    else:
        bits_result = cirq.ResultDict(params=params, measurements={'m': bits})
        digits_result = cirq.ResultDict(params=params, measurements={'m': digits})
    bits_json = cirq.to_json(bits_result)
    digits_json = cirq.to_json(digits_result)
    loaded_bits_result = cirq.read_json(json_text=bits_json)
    loaded_digits_result = cirq.read_json(json_text=digits_json)
    if use_records:
        assert loaded_bits_result.records['m'].dtype == np.uint8
        assert loaded_digits_result.records['m'].dtype == np.uint8
    else:
        assert loaded_bits_result.measurements['m'].dtype == np.uint8
        assert loaded_digits_result.measurements['m'].dtype == np.uint8
    np.testing.assert_allclose(len(bits_json), len(digits_json) / 8, rtol=0.02)