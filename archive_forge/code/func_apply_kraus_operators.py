from typing import Iterable, Sequence
import numpy as np
import pytest
import cirq
def apply_kraus_operators(kraus_operators: Sequence[np.ndarray], rho: np.ndarray) -> np.ndarray:
    d_out, d_in = kraus_operators[0].shape
    assert rho.shape == (d_in, d_in)
    out = np.zeros((d_out, d_out), dtype=np.complex128)
    for k in kraus_operators:
        out += k @ rho @ k.conj().T
    return out