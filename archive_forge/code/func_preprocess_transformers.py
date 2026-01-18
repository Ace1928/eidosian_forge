import itertools
from typing import cast, Any, Dict, List, Optional, Sequence
import cirq
from cirq.protocols.decompose_protocol import DecomposeResult
from cirq_google import ops
from cirq_google.transformers.analytical_decompositions import two_qubit_to_sycamore
@property
def preprocess_transformers(self) -> List[cirq.TRANSFORMER]:
    return [cirq.create_transformer_with_kwargs(cirq.expand_composite, no_decomp=lambda op: cirq.num_qubits(op) <= self.num_qubits), cirq.create_transformer_with_kwargs(merge_swap_rzz_and_2q_unitaries, intermediate_result_tag=self._intermediate_result_tag)]