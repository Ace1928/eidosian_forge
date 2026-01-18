import string
from typing import List, Sequence
from networkx import MultiDiGraph
import pennylane as qml
from pennylane import numpy as pnp
from .utils import MeasureNode, PrepareNode
def qcut_processing_fn(results: Sequence[Sequence], communication_graph: MultiDiGraph, prepare_nodes: Sequence[Sequence[PrepareNode]], measure_nodes: Sequence[Sequence[MeasureNode]], use_opt_einsum: bool=False):
    """Processing function for the :func:`cut_circuit() <pennylane.cut_circuit>` transform.

    .. note::

        This function is designed for use as part of the circuit cutting workflow.
        Check out the :func:`qml.cut_circuit() <pennylane.cut_circuit>` transform for more details.

    Args:
        results (Sequence[Sequence]): A collection of execution results generated from the
            expansion of circuit fragments over measurement and preparation node configurations.
            These results are processed into tensors and then contracted.
        communication_graph (nx.MultiDiGraph): the communication graph determining connectivity
            between circuit fragments
        prepare_nodes (Sequence[Sequence[PrepareNode]]): a sequence of size
            ``len(communication_graph.nodes)`` that determines the order of preparation indices in
            each tensor
        measure_nodes (Sequence[Sequence[MeasureNode]]): a sequence of size
            ``len(communication_graph.nodes)`` that determines the order of measurement indices in
            each tensor
        use_opt_einsum (bool): Determines whether to use the
            `opt_einsum <https://dgasmith.github.io/opt_einsum/>`__ package. This package is useful
            for faster tensor contractions of large networks but must be installed separately using,
            e.g., ``pip install opt_einsum``. Both settings for ``use_opt_einsum`` result in a
            differentiable contraction.

    Returns:
        float or tensor_like: the output of the original uncut circuit arising from contracting
        the tensor network of circuit fragments
    """
    results = [qml.math.stack(tape_res) if isinstance(tape_res, tuple) else qml.math.reshape(tape_res, [-1]) for tape_res in results]
    flat_results = qml.math.concatenate(results)
    tensors = _to_tensors(flat_results, prepare_nodes, measure_nodes)
    result = contract_tensors(tensors, communication_graph, prepare_nodes, measure_nodes, use_opt_einsum)
    return result