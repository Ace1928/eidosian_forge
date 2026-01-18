from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional, List
import rpcq
from qcs_api_client.client import QCSClientConfiguration
from rpcq.messages import TargetDevice as TargetQuantumProcessor
def generate_randomized_benchmarking_sequence(self, request: GenerateRandomizedBenchmarkingSequenceRequest) -> GenerateRandomizedBenchmarkingSequenceResponse:
    """
        Generate a randomized benchmarking sequence.
        """
    rpcq_request = rpcq.messages.RandomizedBenchmarkingRequest(depth=request.depth, qubits=request.num_qubits, gateset=request.gateset, seed=request.seed, interleaver=request.interleaver)
    with self._rpcq_client() as rpcq_client:
        response: rpcq.messages.RandomizedBenchmarkingResponse = rpcq_client.call('generate_rb_sequence', rpcq_request)
        return GenerateRandomizedBenchmarkingSequenceResponse(sequence=response.sequence)