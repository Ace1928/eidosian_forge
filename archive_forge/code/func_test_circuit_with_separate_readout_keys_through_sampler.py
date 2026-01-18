from typing import cast, Tuple
import cirq
import pytest
import sympy
from pyquil import get_qc
from pyquil.api import QVM
from cirq_rigetti import RigettiQCSSampler
@pytest.mark.rigetti_integration
def test_circuit_with_separate_readout_keys_through_sampler(circuit_with_separate_readout_keys: Tuple[cirq.Circuit, cirq.Linspace]) -> None:
    """test that RigettiQCSSampler can properly readout from separate memory
    regions.
    """
    qc = get_qc('9q-square', as_qvm=True)
    sampler = RigettiQCSSampler(quantum_computer=qc)
    qvm = cast(QVM, qc.qam)
    qvm.random_seed = 11
    repetitions = 10
    circuit, sweepable = circuit_with_separate_readout_keys
    results = sampler.run_sweep(program=circuit, params=sweepable, repetitions=repetitions)
    assert len(list(sweepable)) == len(results)
    for i, result in enumerate(results):
        assert isinstance(result, cirq.study.Result)
        assert sweepable[i] == result.params
        assert 'm0' in result.measurements
        assert 'm1' in result.measurements
        assert (repetitions, 1) == result.measurements['m0'].shape
        assert (repetitions, 1) == result.measurements['m1'].shape
        counter0 = result.histogram(key='m0')
        counter1 = result.histogram(key='m1')
        assert 2 == len(counter0)
        assert 3 == counter0.get(0)
        assert 7 == counter0.get(1)
        assert 1 == len(counter1)
        assert 10 == counter1.get(0)
        assert counter1.get(1) is None