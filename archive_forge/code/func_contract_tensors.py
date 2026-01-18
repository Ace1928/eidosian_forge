import string
from typing import List, Sequence
from networkx import MultiDiGraph
import pennylane as qml
from pennylane import numpy as pnp
from .utils import MeasureNode, PrepareNode
def contract_tensors(tensors: Sequence, communication_graph: MultiDiGraph, prepare_nodes: Sequence[Sequence[PrepareNode]], measure_nodes: Sequence[Sequence[MeasureNode]], use_opt_einsum: bool=False):
    """Contract tensors according to the edges specified in the communication graph.

    .. note::

        This function is designed for use as part of the circuit cutting workflow.
        Check out the :func:`qml.cut_circuit() <pennylane.cut_circuit>` transform for more details.

    Consider the three tensors :math:`T^{(1)}`, :math:`T^{(2)}`, and :math:`T^{(3)}`, along with
    their contraction equation

    .. math::

        \\sum_{ijklmn} T^{(1)}_{ij,km} T^{(2)}_{kl,in} T^{(3)}_{mn,jl}

    Each tensor is the result of the tomography of a circuit fragment and has some indices
    corresponding to state preparations (marked by the indices before the comma) and some indices
    corresponding to measurements (marked by the indices after the comma).

    An equivalent representation of the contraction equation is to use a directed multigraph known
    as the communication/quotient graph. In the communication graph, each tensor is assigned a node
    and edges are added between nodes to mark a contraction along an index. The communication graph
    resulting from the above contraction equation is a complete directed graph.

    In the communication graph provided by :func:`fragment_graph`, edges are composed of
    :class:`PrepareNode` and :class:`MeasureNode` pairs. To correctly map back to the contraction
    equation, we must keep track of the order of preparation and measurement indices in each tensor.
    This order is specified in the ``prepare_nodes`` and ``measure_nodes`` arguments.

    Args:
        tensors (Sequence): the tensors to be contracted
        communication_graph (nx.MultiDiGraph): the communication graph determining connectivity
            between the tensors
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
        float or tensor_like: the result of contracting the tensor network

    **Example**

    We first set up the tensors and their corresponding :class:`~.PrepareNode` and
    :class:`~.MeasureNode` orderings:

    .. code-block:: python

        from pennylane.transforms import qcut
        import networkx as nx
        import numpy as np

        tensors = [np.arange(4), np.arange(4, 8)]
        prep = [[], [qcut.PrepareNode(wires=0)]]
        meas = [[qcut.MeasureNode(wires=0)], []]

    The communication graph describing edges in the tensor network must also be constructed.
    The nodes of the fragment graphs are formatted as ``WrappedObj(op)``, where ``WrappedObj.obj``
    is the operator, and the same format should be preserved in the pairs stored
    with the edge data of the communication graph:

    .. code-block:: python

        graph = nx.MultiDiGraph(
            [(0, 1, {"pair": (WrappedObj(meas[0][0]), WrappedObj(prep[1][0]))})]
        )

    The network can then be contracted using:

    >>> qml.qcut.contract_tensors(tensors, graph, prep, meas)
    38
    """
    if use_opt_einsum:
        try:
            from opt_einsum import contract, get_symbol
        except ImportError as e:
            raise ImportError('The opt_einsum package is required when use_opt_einsum is set to True in the contract_tensors function. This package can be installed using:\npip install opt_einsum') from e
    else:
        contract = qml.math.einsum
        get_symbol = _get_symbol
    ctr = 0
    tensor_indxs = [''] * len(communication_graph.nodes)
    meas_map = {}
    for i, (node, prep) in enumerate(zip(communication_graph.nodes, prepare_nodes)):
        predecessors = communication_graph.pred[node]
        for p in prep:
            for _, pred_edges in predecessors.items():
                for pred_edge in pred_edges.values():
                    meas_op, prep_op = pred_edge['pair']
                    if p.id is prep_op.obj.id:
                        symb = get_symbol(ctr)
                        ctr += 1
                        tensor_indxs[i] += symb
                        meas_map[meas_op] = symb
    for i, (node, meas) in enumerate(zip(communication_graph.nodes, measure_nodes)):
        successors = communication_graph.succ[node]
        for m in meas:
            for _, succ_edges in successors.items():
                for succ_edge in succ_edges.values():
                    meas_op, _ = succ_edge['pair']
                    if m.id is meas_op.obj.id:
                        symb = meas_map[meas_op]
                        tensor_indxs[i] += symb
    eqn = ','.join(tensor_indxs)
    kwargs = {} if use_opt_einsum else {'like': tensors[0]}
    return contract(eqn, *tensors, **kwargs)