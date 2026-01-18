from typing import Optional, TYPE_CHECKING
from cirq.transformers import transformer_api, transformer_primitives
@transformer_api.transformer
def drop_empty_moments(circuit: 'cirq.AbstractCircuit', *, context: Optional['cirq.TransformerContext']=None) -> 'cirq.Circuit':
    """Removes empty moments from a circuit.

    Args:
          circuit: Input circuit to transform.
          context: `cirq.TransformerContext` storing common configurable options for transformers.

    Returns:
          Copy of the transformed input circuit.
    """
    if context is None:
        context = transformer_api.TransformerContext()
    return transformer_primitives.map_moments(circuit.unfreeze(False), lambda m, _: m if m else [], deep=context.deep, tags_to_ignore=context.tags_to_ignore)