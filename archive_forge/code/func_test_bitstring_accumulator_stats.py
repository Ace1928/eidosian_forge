import dataclasses
import datetime
import time
import numpy as np
import pytest
import cirq
import cirq.work as cw
from cirq.work.observable_measurement_data import (
from cirq.work.observable_settings import _MeasurementSpec
def test_bitstring_accumulator_stats():
    kwargs = _get_ZZ_Z_Z_bsa_constructor_args()
    settings = kwargs['simul_settings']
    a, b = kwargs['qubit_to_index']
    bsa = cw.BitstringAccumulator(**kwargs)
    np.testing.assert_allclose([0, 0, 0], bsa.means())
    should_be = np.array([[4 * 7 ** 2, 0, 0], [0, 4 * 5 ** 2, 0], [0, 0, 4 * 3 ** 2]])
    should_be = should_be / (4 - 1)
    should_be = should_be / 4
    np.testing.assert_allclose(should_be, bsa.covariance())
    for setting, var in zip(settings, [4 * 7 ** 2, 4 * 5 ** 2, 4 * 3 ** 2]):
        np.testing.assert_allclose(0, bsa.mean(setting))
        np.testing.assert_allclose(var / 4 / (4 - 1), bsa.variance(setting))
        np.testing.assert_allclose(np.sqrt(var / 4 / (4 - 1)), bsa.stderr(setting))
    bad_obs = [cirq.X(a) * cirq.X(b)]
    bad_setting = list(cw.observables_to_settings(bad_obs, qubits=[a, b]))[0]
    with pytest.raises(ValueError):
        bsa.mean(bad_setting)