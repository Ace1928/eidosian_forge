from typing import Optional, List, Hashable, TYPE_CHECKING
import abc
from cirq import circuits, ops, protocols, transformers
from cirq.protocols.decompose_protocol import DecomposeResult
from cirq.transformers import merge_k_qubit_gates, merge_single_qubit_gates
def create_transformer_with_kwargs(transformer: 'cirq.TRANSFORMER', **kwargs) -> 'cirq.TRANSFORMER':
    """Method to capture additional keyword arguments to transformers while preserving mypy type.

    Returns a `cirq.TRANSFORMER` which, when called with a circuit and transformer context, is
    equivalent to calling `transformer(circuit, context=context, **kwargs)`. It is often useful to
    capture keyword arguments of a transformer before passing them as an argument to an API that
    expects `cirq.TRANSFORMER`. For example:

    >>> def run_transformers(transformers: 'List[cirq.TRANSFORMER]'):
    ...     circuit = cirq.Circuit(cirq.X(cirq.q(0)))
    ...     context = cirq.TransformerContext()
    ...     for transformer in transformers:
    ...         transformer(circuit, context=context)
    ...
    >>> transformers: 'List[cirq.TRANSFORMER]' = []
    >>> transformers.append(
    ...     cirq.create_transformer_with_kwargs(
    ...         cirq.expand_composite, no_decomp=lambda op: cirq.num_qubits(op) <= 2
    ...     )
    ... )
    >>> transformers.append(cirq.create_transformer_with_kwargs(cirq.merge_k_qubit_unitaries, k=2))
    >>> run_transformers(transformers)


    Args:
         transformer: A `cirq.TRANSFORMER` for which additional kwargs should be captured.
         **kwargs: The keyword arguments which should be captured and passed to `transformer`.

    Returns:
        A `cirq.TRANSFORMER` method `transformer_with_kwargs`, s.t. executing
        `transformer_with_kwargs(circuit, context=context)` is equivalent to executing
        `transformer(circuit, context=context, **kwargs)`.

    Raises:
        SyntaxError: if **kwargs contain a 'context'.
    """
    if 'context' in kwargs:
        raise SyntaxError('**kwargs to be captured must not contain `context`.')

    def transformer_with_kwargs(circuit: 'cirq.AbstractCircuit', *, context: Optional['cirq.TransformerContext']=None) -> 'cirq.AbstractCircuit':
        return transformer(circuit, context=context, **kwargs)
    return transformer_with_kwargs