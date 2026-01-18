import dataclasses
import datetime
import time
import numpy as np
import pytest
import cirq
import cirq.work as cw
from cirq.work.observable_measurement_data import (
from cirq.work.observable_settings import _MeasurementSpec
def test_bitstring_accumulator(example_bsa):
    assert example_bsa.bitstrings.shape == (0, 2)
    assert example_bsa.chunksizes.shape == (0,)
    assert example_bsa.timestamps.shape == (0,)
    bitstrings = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.uint8)
    example_bsa.consume_results(bitstrings)
    assert example_bsa.bitstrings.shape == (4, 2)
    assert example_bsa.chunksizes.shape == (1,)
    assert example_bsa.timestamps.shape == (1,)
    assert example_bsa.n_repetitions == 4
    with pytest.raises(ValueError):
        example_bsa.consume_results(bitstrings.astype(int))
    results = list(example_bsa.results)
    assert len(results) == 3
    for r in results:
        assert r.repetitions == 4
    for r in example_bsa.records:
        assert isinstance(r, dict)
        assert 'repetitions' in r
        assert r['repetitions'] == 4