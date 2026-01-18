from dataclasses import dataclass
from typing import Optional, List, Callable, Dict, Tuple, Set, Any
import networkx as nx
import numpy as np
import pandas as pd
import cirq
import cirq.contrib.routing as ccr
def compile_circuit(circuit: cirq.Circuit, *, device_graph: nx.Graph, routing_attempts: int, compiler: Optional[Callable[[cirq.Circuit], cirq.Circuit]]=None, routing_algo_name: Optional[str]=None, router: Optional[Callable[..., ccr.SwapNetwork]]=None, add_readout_error_correction=False) -> CompilationResult:
    """Compile the given model circuit onto the given device graph. This uses a
    different compilation method than described in
    https://arxiv.org/pdf/1811.12926.pdf Appendix A. The latter goes through a
    7-step process involving various decompositions, routing, and optimization
    steps. We route the model circuit and then run a series of optimizers on it
    (which can be passed into this function).

    Args:
        circuit: The model circuit to compile.
        device_graph: The device graph to compile onto.
        routing_attempts: See doc for calculate_quantum_volume.
        compiler: An optional function to deconstruct the model circuit's
            gates down to the target devices gate set and then optimize it.
        routing_algo_name: The name of the routing algorithm, see ROUTING in
            `route_circuit.py`.
        router: The function that actually does the routing.
        add_readout_error_correction: If true, add some parity bits that will
            later be used to detect readout error.

    Returns: A tuple where the first value is the compiled circuit and the
        second value is the final mapping from the model circuit to the compiled
        circuit. The latter is necessary in order to preserve the measurement
        order.

    """
    compiled_circuit = circuit.copy()
    parity_map: Dict[cirq.Qid, cirq.Qid] = {}
    if add_readout_error_correction:
        num_qubits = len(compiled_circuit.all_qubits())
        for idx, qubit in enumerate(sorted(compiled_circuit.all_qubits())):
            qubit_num = idx + num_qubits
            parity_qubit = cirq.LineQubit(qubit_num)
            compiled_circuit.append(cirq.X(qubit))
            compiled_circuit.append(cirq.CNOT(qubit, parity_qubit))
            compiled_circuit.append(cirq.X(qubit))
            parity_map[qubit] = parity_qubit
    if router is None and routing_algo_name is None:
        routing_algo_name = 'greedy'
    swap_networks: List[ccr.SwapNetwork] = []
    for _ in range(routing_attempts):
        swap_network = ccr.route_circuit(compiled_circuit, device_graph, router=router, algo_name=routing_algo_name)
        swap_networks.append(swap_network)
    assert len(swap_networks) > 0, 'Unable to get routing for circuit'
    swap_networks.sort(key=lambda swap_network: (len(swap_network.circuit.all_qubits()), len(swap_network.circuit)))
    routed_circuit = swap_networks[0].circuit
    mapping = swap_networks[0].final_mapping()

    def replace_swap_permutation_gate(op: 'cirq.Operation', _):
        if isinstance(op.gate, cirq.contrib.acquaintance.SwapPermutationGate):
            return [op.gate.swap_gate.on(*op.qubits)]
        return op
    routed_circuit = cirq.map_operations_and_unroll(routed_circuit, map_func=replace_swap_permutation_gate)
    if not compiler:
        return CompilationResult(circuit=routed_circuit, mapping=mapping, parity_map=parity_map)
    return CompilationResult(circuit=compiler(routed_circuit), mapping=mapping, parity_map=parity_map)