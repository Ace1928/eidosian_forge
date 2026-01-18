from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional, List
import rpcq
from qcs_api_client.client import QCSClientConfiguration
from rpcq.messages import TargetDevice as TargetQuantumProcessor
def conjugate_pauli_by_clifford(self, request: ConjugatePauliByCliffordRequest) -> ConjugatePauliByCliffordResponse:
    """
        Conjugate a Pauli element by a Clifford element.
        """
    rpcq_request = rpcq.messages.ConjugateByCliffordRequest(pauli=rpcq.messages.PauliTerm(indices=request.pauli_indices, symbols=request.pauli_symbols), clifford=request.clifford)
    with self._rpcq_client() as rpcq_client:
        response: rpcq.messages.ConjugateByCliffordResponse = rpcq_client.call('conjugate_pauli_by_clifford', rpcq_request)
        return ConjugatePauliByCliffordResponse(phase_factor=response.phase, pauli=response.pauli)