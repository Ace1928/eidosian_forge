import uuid
from typing import Any, Callable, Sequence, Tuple
import warnings
import numpy as np
from networkx import MultiDiGraph, has_path, weakly_connected_components
import pennylane as qml
from pennylane.measurements import MeasurementProcess
from pennylane.ops.meta import WireCut
from pennylane.queuing import WrappedObj
from pennylane.operation import Operation
from .kahypar import kahypar_cut
from .cutstrategy import CutStrategy
def find_and_place_cuts(graph: MultiDiGraph, cut_method: Callable=kahypar_cut, cut_strategy: CutStrategy=None, replace_wire_cuts=False, local_measurement=False, **kwargs) -> MultiDiGraph:
    """Automatically finds and places optimal :class:`~.WireCut` nodes into a given tape-converted graph
    using a customizable graph partitioning function. Preserves existing placed cuts.

    Args:
        graph (MultiDiGraph): The original (tape-converted) graph to be cut.
        cut_method (Callable): A graph partitioning function that takes an input graph and returns
            a list of edges to be cut based on a given set of constraints and objective. Defaults
            to :func:`kahypar_cut` which requires KaHyPar to be installed using
            ``pip install kahypar`` for Linux and Mac users or visiting the
            instructions `here <https://kahypar.org>`__ to compile from
            source for Windows users.
        cut_strategy (CutStrategy): Strategy for optimizing cutting parameters based on device
            constraints. Defaults to ``None`` in which case ``kwargs`` must be fully specified
            for passing to the ``cut_method``.
        replace_wire_cuts (bool): Whether to replace :class:`~.WireCut` nodes with
            :class:`~.MeasureNode` and :class:`~.PrepareNode` pairs. Defaults to ``False``.
        local_measurement (bool): Whether to use the local-measurement circuit-cutting objective,
            i.e. the maximum node-degree of the communication graph, for cut evaluation. Defaults
            to ``False`` which assumes global measurement and uses the total number of cuts as the
            cutting objective.
        kwargs: Additional keyword arguments to be passed to the callable ``cut_method``.

    Returns:
        nx.MultiDiGraph: Copy of the input graph with :class:`~.WireCut` nodes inserted.

    **Example**

    Consider the following 4-wire circuit with a single CNOT gate connecting the top (wires
    ``[0, 1]``) and bottom (wires ``["a", "b"]``) halves of the circuit. Note there's a
    :class:`~.WireCut` manually placed into the circuit already.

    .. code-block:: python

        ops = [
            qml.RX(0.1, wires=0),
            qml.RY(0.2, wires=1),
            qml.RX(0.3, wires="a"),
            qml.RY(0.4, wires="b"),
            qml.CNOT(wires=[0, 1]),
            qml.WireCut(wires=1),
            qml.CNOT(wires=["a", "b"]),
            qml.CNOT(wires=[1, "a"]),
            qml.CNOT(wires=[0, 1]),
            qml.CNOT(wires=["a", "b"]),
            qml.RX(0.5, wires="a"),
            qml.RY(0.6, wires="b"),
        ]
        measurements = [qml.expval(qml.X(0) @ qml.Y("a") @ qml.Z("b"))]
        tape = qml.tape.QuantumTape(ops, measurements)

    >>> print(qml.drawer.tape.text(tape))
     0: ──RX(0.1)──╭●──────────╭●───────────╭┤ ⟨X ⊗ Y ⊗ Z⟩
     1: ──RY(0.2)──╰X──//──╭●──╰X───────────│┤
     a: ──RX(0.3)──╭●──────╰X──╭●──RX(0.5)──├┤ ⟨X ⊗ Y ⊗ Z⟩
     b: ──RY(0.4)──╰X──────────╰X──RY(0.6)──╰┤ ⟨X ⊗ Y ⊗ Z⟩

    Since the existing :class:`~.WireCut` doesn't sufficiently fragment the circuit, we can find the
    remaining cuts using the default KaHyPar partitioner:

    >>> graph = qml.qcut.tape_to_graph(tape)
    >>> cut_graph = qml.qcut.find_and_place_cuts(
            graph=graph,
            num_fragments=2,
            imbalance=0.5,
        )

    Visualizing the newly-placed cut:

    >>> print(qml.qcut.graph_to_tape(cut_graph).draw())
     0: ──RX(0.1)──╭●───────────────╭●────────╭┤ ⟨X ⊗ Y ⊗ Z⟩
     1: ──RY(0.2)──╰X──//──╭●───//──╰X────────│┤
     a: ──RX(0.3)──╭●──────╰X──╭●────RX(0.5)──├┤ ⟨X ⊗ Y ⊗ Z⟩
     b: ──RY(0.4)──╰X──────────╰X────RY(0.6)──╰┤ ⟨X ⊗ Y ⊗ Z⟩

    We can then proceed with the usual process of replacing :class:`~.WireCut` nodes with
    pairs of :class:`~.MeasureNode` and :class:`~.PrepareNode`, and then break the graph
    into fragments. Or, alternatively, we can directly get such processed graph by passing
    ``replace_wire_cuts=True``:

    >>> cut_graph = qml.qcut.find_and_place_cuts(
            graph=graph,
            num_fragments=2,
            imbalance=0.5,
            replace_wire_cuts=True,
        )
    >>> frags, comm_graph = qml.qcut.fragment_graph(cut_graph)
    >>> for t in frags:
    ...     print(qml.qcut.graph_to_tape(t).draw())

    .. code-block::

         0: ──RX(0.1)──────╭●───────────────╭●──┤ ⟨X⟩
         1: ──RY(0.2)──────╰X──MeasureNode──│───┤
         2: ──PrepareNode───────────────────╰X──┤

         a: ──RX(0.3)──────╭●──╭X──╭●────────────RX(0.5)──╭┤ ⟨Y ⊗ Z⟩
         b: ──RY(0.4)──────╰X──│───╰X────────────RY(0.6)──╰┤ ⟨Y ⊗ Z⟩
         1: ──PrepareNode──────╰●───MeasureNode────────────┤

    Alternatively, if all we want to do is to find the optimal way to fit a circuit onto a smaller
    device, a :class:`~.CutStrategy` can be used to populate the necessary explorations of cutting
    parameters. As an extreme example, if the only device at our disposal is a 2-qubit device, a
    simple cut strategy is to simply specify the the ``max_free_wires`` argument (or equivalently
    directly passing a :class:`pennylane.Device` to the ``device`` argument):

    >>> cut_strategy = qml.qcut.CutStrategy(max_free_wires=2)
    >>> print(cut_strategy.get_cut_kwargs(graph))
     [{'num_fragments': 2, 'imbalance': 0.5714285714285714},
      {'num_fragments': 3, 'imbalance': 1.4},
      {'num_fragments': 4, 'imbalance': 1.75},
      {'num_fragments': 5, 'imbalance': 2.3333333333333335},
      {'num_fragments': 6, 'imbalance': 2.0},
      {'num_fragments': 7, 'imbalance': 3.0},
      {'num_fragments': 8, 'imbalance': 2.5},
      {'num_fragments': 9, 'imbalance': 2.0},
      {'num_fragments': 10, 'imbalance': 1.5},
      {'num_fragments': 11, 'imbalance': 1.0},
      {'num_fragments': 12, 'imbalance': 0.5},
      {'num_fragments': 13, 'imbalance': 0.05},
      {'num_fragments': 14, 'imbalance': 0.1}]

    The printed list above shows all the possible cutting configurations one can attempt to perform
    in order to search for the optimal cut. This is done by directly passing a
    :class:`~.CutStrategy` to :func:`~.find_and_place_cuts`:

    >>> cut_graph = qml.qcut.find_and_place_cuts(
            graph=graph,
            cut_strategy=cut_strategy,
        )
    >>> print(qml.qcut.graph_to_tape(cut_graph).draw())
     0: ──RX──//─╭●──//────────╭●──//─────────┤ ╭<X@Y@Z>
     1: ──RY──//─╰X──//─╭●──//─╰X─────────────┤ │
     a: ──RX──//─╭●──//─╰X──//─╭●──//──RX──//─┤ ├<X@Y@Z>
     b: ──RY──//─╰X──//────────╰X──//──RY─────┤ ╰<X@Y@Z>

    As one can tell, quite a few cuts have to be made in order to execute the circuit on solely
    2-qubit devices. To verify, let's print the fragments:

    >>> qml.qcut.replace_wire_cut_nodes(cut_graph)
    >>> frags, comm_graph = qml.qcut.fragment_graph(cut_graph)
    >>> for t in frags:
    ...     print(qml.qcut.graph_to_tape(t).draw())

    .. code-block::

         0: ──RX──MeasureNode─┤

         1: ──RY──MeasureNode─┤

         a: ──RX──MeasureNode─┤

         b: ──RY──MeasureNode─┤

         0: ──PrepareNode─╭●──MeasureNode─┤
         1: ──PrepareNode─╰X──MeasureNode─┤

         a: ──PrepareNode─╭●──MeasureNode─┤
         b: ──PrepareNode─╰X──MeasureNode─┤

         1: ──PrepareNode─╭●──MeasureNode─┤
         a: ──PrepareNode─╰X──MeasureNode─┤

         0: ──PrepareNode─╭●──MeasureNode─┤
         1: ──PrepareNode─╰X──────────────┤

         b: ──PrepareNode─╭X──MeasureNode─┤
         a: ──PrepareNode─╰●──MeasureNode─┤

         a: ──PrepareNode──RX──MeasureNode─┤

         b: ──PrepareNode──RY─┤  <Z>

         0: ──PrepareNode─┤  <X>

         a: ──PrepareNode─┤  <Y>

    """
    cut_graph = _remove_existing_cuts(graph)
    if isinstance(cut_strategy, CutStrategy):
        cut_kwargs_probed = cut_strategy.get_cut_kwargs(cut_graph)
        seed = kwargs.pop('seed', None)
        seeds = np.random.default_rng(seed).choice(2 ** 15, cut_strategy.trials_per_probe).tolist()
        cut_edges_probed = {(cut_kwargs['num_fragments'], trial_id): cut_method(cut_graph, **{**cut_kwargs, **kwargs, 'seed': seed}) for cut_kwargs in cut_kwargs_probed for trial_id, seed in zip(range(cut_strategy.trials_per_probe), seeds)}
        valid_cut_edges = {}
        for (num_partitions, _), cut_edges in cut_edges_probed.items():
            cut_graph = place_wire_cuts(graph=graph, cut_edges=cut_edges)
            num_cuts = sum((isinstance(n.obj, WireCut) for n in cut_graph.nodes))
            replace_wire_cut_nodes(cut_graph)
            frags, comm = fragment_graph(cut_graph)
            max_frag_degree = max(dict(comm.degree()).values())
            if _is_valid_cut(fragments=frags, num_cuts=num_cuts, max_frag_degree=max_frag_degree, num_fragments_requested=num_partitions, cut_candidates=valid_cut_edges, max_free_wires=cut_strategy.max_free_wires):
                key = (len(frags), max_frag_degree)
                valid_cut_edges[key] = cut_edges
        if len(valid_cut_edges) < 1:
            raise ValueError('Unable to find a circuit cutting that satisfies all constraints. Are the constraints too strict?')
        cut_edges = _get_optim_cut(valid_cut_edges, local_measurement=local_measurement)
    else:
        cut_edges = cut_method(cut_graph, **kwargs)
    cut_graph = place_wire_cuts(graph=graph, cut_edges=cut_edges)
    if replace_wire_cuts:
        replace_wire_cut_nodes(cut_graph)
    return cut_graph