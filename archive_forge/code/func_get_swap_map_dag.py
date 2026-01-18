from qiskit.transpiler.exceptions import TranspilerError
from .algorithms import ApproximateTokenSwapper
def get_swap_map_dag(dag, coupling_map, from_layout, to_layout, seed, trials=4):
    """Get the circuit of swaps to go from from_layout to to_layout, and the physical qubits
    (integers) that the swap circuit should be applied on."""
    if len(dag.qregs) != 1 or dag.qregs.get('q', None) is None:
        raise TranspilerError('layout transformation runs on physical circuits only')
    if len(dag.qubits) > len(coupling_map.physical_qubits):
        raise TranspilerError('The layout does not match the amount of qubits in the DAG')
    token_swapper = ApproximateTokenSwapper(coupling_map.graph.to_undirected(), seed)
    permutation = {pqubit: to_layout[vqubit] for vqubit, pqubit in from_layout.get_virtual_bits().items()}
    swap_circuit, phys_to_circuit_qubits = token_swapper.permutation_circuit(permutation, trials)
    circuit_to_phys = {inner: outer for outer, inner in phys_to_circuit_qubits.items()}
    return (swap_circuit, [circuit_to_phys[bit] for bit in swap_circuit.qubits])