from __future__ import annotations
from collections import defaultdict
from qiskit.circuit import Gate, QuantumCircuit, Qubit
from qiskit.converters import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit, DAGOpNode
from qiskit.transpiler import TransformationPass, Layout, TranspilerError
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing.swap_strategy import SwapStrategy
from qiskit.transpiler.passes.routing.commuting_2q_gate_routing.commuting_2q_block import (
def swap_decompose(self, dag: DAGCircuit, node: DAGOpNode, current_layout: Layout, swap_strategy: SwapStrategy) -> DAGCircuit:
    """Take an instance of :class:`.Commuting2qBlock` and map it to the coupling map.

        The mapping is done with the swap strategy.

        Args:
            dag: The dag which contains the :class:`.Commuting2qBlock` we route.
            node: A node whose operation is a :class:`.Commuting2qBlock`.
            current_layout: The layout before the swaps are applied. This function will
                modify the layout so that subsequent gates can be properly composed on the dag.
            swap_strategy: The swap strategy used to decompose the node.

        Returns:
            A dag that is compatible with the coupling map where swap gates have been added
            to map the gates in the :class:`.Commuting2qBlock` to the hardware.
        """
    trivial_layout = Layout.generate_trivial_layout(*dag.qregs.values())
    gate_layers = self._make_op_layers(dag, node.op, current_layout, swap_strategy)
    max_distance = max(gate_layers.keys())
    circuit_with_swap = QuantumCircuit(len(dag.qubits))
    for i in range(max_distance + 1):
        current_layer = {}
        for (j, k), local_gate in gate_layers.get(i, {}).items():
            current_layer[self._position_in_cmap(dag, j, k, current_layout)] = local_gate
        sub_layers = self._build_sub_layers(current_layer)
        for sublayer in sub_layers:
            for edge, local_gate in sublayer.items():
                circuit_with_swap.append(local_gate, edge)
        if i < max_distance:
            for swap in swap_strategy.swap_layer(i):
                j, k = [trivial_layout.get_physical_bits()[vertex] for vertex in swap]
                circuit_with_swap.swap(j, k)
                current_layout.swap(j, k)
    return circuit_to_dag(circuit_with_swap)