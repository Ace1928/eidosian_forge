from typing import Optional, Sequence, Tuple
import cirq
import cirq_ft
import numpy as np
import pytest
from attr import frozen
from cirq_ft import infra
from cirq._compat import cached_property
from cirq_ft.algos.mean_estimation import CodeForRandomVariable, MeanEstimationOperator
from cirq_ft.infra import bit_tools
from cirq_ft.deprecation import allow_deprecated_cirq_ft_use_in_tests
@pytest.mark.parametrize('n, marked_val, c', [(5, 1, 4), (4, 1, 2), (2, 1, np.sqrt(2))])
@allow_deprecated_cirq_ft_use_in_tests
def test_mean_estimation_grover(n: int, marked_val: int, c: float, marked_item: int=1, arctan_bitsize: int=5):
    synthesizer = GroverSynthesizer(n)
    encoder = GroverEncoder(n, marked_item=marked_item, marked_val=marked_val)
    s = np.sqrt(encoder.s_square)
    assert c * s < 1 and c >= 1 >= s
    assert satisfies_theorem_321(synthesizer=synthesizer, encoder=encoder, c=c, s=s, mu=encoder.mu, arctan_bitsize=arctan_bitsize)