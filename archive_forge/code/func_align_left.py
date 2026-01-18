import dataclasses
from typing import Optional, TYPE_CHECKING
from cirq import circuits, ops
from cirq.transformers import transformer_api
@transformer_api.transformer(add_deep_support=True)
def align_left(circuit: 'cirq.AbstractCircuit', *, context: Optional['cirq.TransformerContext']=None) -> 'cirq.Circuit':
    """Align gates to the left of the circuit.

    Note that tagged operations with tag in `context.tags_to_ignore` will continue to stay in their
    original position and will not be aligned.

    Args:
          circuit: Input circuit to transform.
          context: `cirq.TransformerContext` storing common configurable options for transformers.

    Returns:
          Copy of the transformed input circuit.
    """
    if context is None:
        context = transformer_api.TransformerContext()
    ret = circuits.Circuit()
    for i, moment in enumerate(circuit):
        for op in moment:
            if isinstance(op, ops.TaggedOperation) and set(op.tags).intersection(context.tags_to_ignore):
                ret.append([circuits.Moment()] * (i + 1 - len(ret)))
                ret[i] = ret[i].with_operation(op)
            else:
                ret.append(op)
    return ret