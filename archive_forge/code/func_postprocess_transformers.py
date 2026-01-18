from typing import Any, Dict, List, Sequence, Type, Union
import cirq
@property
def postprocess_transformers(self) -> List[cirq.TRANSFORMER]:
    """List of transformers which should be run after decomposing individual operations.

        If `eject_paulis` is enabled in the constructor, adds `cirq.eject_phased_paulis` and
        `cirq.eject_z` in addition to postprocess_transformers already available in
        `cirq.CompilationTargetGateset`.
        """
    transformers: List[cirq.TRANSFORMER] = [cirq.create_transformer_with_kwargs(cirq.merge_single_qubit_moments_to_phxz, atol=self.atol), cirq.create_transformer_with_kwargs(cirq.drop_negligible_operations, atol=self.atol), cirq.drop_empty_moments]
    if self.eject_paulis:
        return transformers[:1] + [cirq.create_transformer_with_kwargs(cirq.eject_phased_paulis, atol=self.atol), cirq.create_transformer_with_kwargs(cirq.eject_z, atol=self.atol)] + transformers[1:]
    return transformers