from collections.abc import Sequence as SequenceType
from itertools import compress
from pathlib import Path
from typing import Any, List, Sequence, Tuple, Union
import numpy as np
from networkx import MultiDiGraph
import pennylane as qml
from pennylane.operation import Operation
def kahypar_cut(graph: MultiDiGraph, num_fragments: int, imbalance: int=None, edge_weights: List[Union[int, float]]=None, node_weights: List[Union[int, float]]=None, fragment_weights: List[Union[int, float]]=None, hyperwire_weight: int=1, seed: int=None, config_path: Union[str, Path]=None, trial: int=None, verbose: bool=False) -> List[Tuple[Operation, Operation, Any]]:
    """Calls `KaHyPar <https://kahypar.org/>`__ to partition a graph.

    .. warning::
        Requires KaHyPar to be installed separately. For Linux and Mac users,
        KaHyPar can be installed using ``pip install kahypar``. Windows users
        can follow the instructions
        `here <https://kahypar.org>`__ to compile from source.

    Args:
        graph (nx.MultiDiGraph): The graph to be partitioned.
        num_fragments (int): Desired number of fragments.
        imbalance (int): Imbalance factor of the partitioning. Defaults to KaHyPar's determination.
        edge_weights (List[Union[int, float]]): Weights for edges. Defaults to unit-weighted edges.
        node_weights (List[Union[int, float]]): Weights for nodes. Defaults to unit-weighted nodes.
        fragment_weights (List[Union[int, float]]): Maximum size constraints by fragment. Defaults
            to no such constraints, with ``imbalance`` the only parameter affecting fragment sizes.
        hyperwire_weight (int): Weight on the artificially appended hyperedges representing wires.
            Setting it to 0 leads to no such insertion. If greater than 0, hyperedges will be
            appended with the provided weight, to encourage the resulting fragments to cluster gates
            on the same wire together. Defaults to 1.
        seed (int): KaHyPar's seed. Defaults to the seed in the config file which defaults to -1,
            i.e. unfixed seed.
        config_path (str): KaHyPar's ``.ini`` config file path. Defaults to its SEA20 paper config.
        trial (int): trial id for summary label creation. Defaults to ``None``.
        verbose (bool): Flag for printing KaHyPar's output summary. Defaults to ``False``.

    Returns:
        List[Union[int, Any]]: List of cut edges.

    **Example**

    Consider the following 2-wire circuit with one CNOT gate connecting the wires:

    .. code-block:: python

        ops = [
            qml.RX(0.432, wires=0),
            qml.RY(0.543, wires="a"),
            qml.CNOT(wires=[0, "a"]),
            qml.RZ(0.240, wires=0),
            qml.RZ(0.133, wires="a"),
            qml.RX(0.432, wires=0),
            qml.RY(0.543, wires="a"),
        ]
        measurements = [qml.expval(qml.Z(0))]
        tape = qml.tape.QuantumTape(ops, measurements)

    We can let KaHyPar automatically find the optimal edges to place cuts:

    >>> graph = qml.qcut.tape_to_graph(tape)
    >>> cut_edges = qml.qcut.kahypar_cut(
            graph=graph,
            num_fragments=2,
        )
    >>> cut_edges
    [(CNOT(wires=[0, 'a']), RZ(0.24, wires=[0]), 0)]
    """
    try:
        import kahypar
    except ImportError as e:
        raise ImportError('KaHyPar must be installed to use this method for automatic cut placement. Try pip install kahypar or visit https://kahypar.org/ for installation instructions.') from e
    adjacent_nodes, edge_splits, edge_weights = _graph_to_hmetis(graph=graph, hyperwire_weight=hyperwire_weight, edge_weights=edge_weights)
    trial = 0 if trial is None else trial
    ne = len(edge_splits) - 1
    nv = max(adjacent_nodes) + 1
    if edge_weights is not None or node_weights is not None:
        edge_weights = edge_weights or [1] * ne
        node_weights = node_weights or [1] * nv
        hypergraph = kahypar.Hypergraph(nv, ne, edge_splits, adjacent_nodes, num_fragments, edge_weights, node_weights)
    else:
        hypergraph = kahypar.Hypergraph(nv, ne, edge_splits, adjacent_nodes, num_fragments)
    context = kahypar.Context()
    config_path = config_path or str(Path(__file__).parent / '_cut_kKaHyPar_sea20.ini')
    context.loadINIconfiguration(config_path)
    context.setK(num_fragments)
    if isinstance(imbalance, float):
        context.setEpsilon(imbalance)
    if isinstance(fragment_weights, SequenceType) and len(fragment_weights) == num_fragments:
        context.setCustomTargetBlockWeights(fragment_weights)
    if not verbose:
        context.suppressOutput(True)
    kahypar_seed = np.random.default_rng(seed).choice(2 ** 15)
    context.setSeed(kahypar_seed)
    kahypar.partition(hypergraph, context)
    cut_edge_mask = [hypergraph.connectivity(e) > 1 for e in hypergraph.edges()]
    cut_edges = list(compress(graph.edges, cut_edge_mask))
    if verbose:
        fragment_sizes = [hypergraph.blockSize(p) for p in range(num_fragments)]
        print(len(fragment_sizes), fragment_sizes)
    return cut_edges