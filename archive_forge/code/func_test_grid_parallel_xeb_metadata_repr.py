from cirq.experiments.grid_parallel_two_qubit_xeb import GridParallelXEBMetadata, LAYER_A, LAYER_B
import cirq
def test_grid_parallel_xeb_metadata_repr():
    metadata = GridParallelXEBMetadata(qubits=cirq.GridQubit.square(2), two_qubit_gate=cirq.ISWAP, num_circuits=10, repetitions=10000, cycles=[2, 4, 6, 8, 10], layers=[LAYER_A, LAYER_B], seed=1234)
    cirq.testing.assert_equivalent_repr(metadata)