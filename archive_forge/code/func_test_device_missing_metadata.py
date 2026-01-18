import itertools
import pytest
import cirq
import cirq_google as cg
import numpy as np
def test_device_missing_metadata():

    class BadDevice(cirq.Device):
        pass
    topo = cirq.TiltedSquareLattice(3, 3)
    qubits = sorted(topo.nodes_to_gridqubits().values())
    circuit = cirq.experiments.random_rotations_between_grid_interaction_layers_circuit(qubits, depth=8, two_qubit_op_factory=lambda a, b, _: cirq.SQRT_ISWAP(a, b))
    qp = cg.RandomDevicePlacer()
    with pytest.raises(ValueError):
        qp.place_circuit(circuit, problem_topology=topo, shared_rt_info=cg.SharedRuntimeInfo(run_id='1', device=BadDevice()), rs=np.random.RandomState(1))