import itertools
import pytest
import cirq
import cirq_google as cg
import numpy as np
def test_hardcoded_qubit_placer():
    rainbow_record = cg.SimulatedProcessorWithLocalDeviceRecord('rainbow')
    rainbow_device = rainbow_record.get_device()
    rainbow_graph = rainbow_device.metadata.nx_graph
    hardcoded = cg.HardcodedQubitPlacer(_all_offset_placements(rainbow_graph))
    topo = cirq.TiltedSquareLattice(3, 2)
    circuit = cirq.experiments.random_rotations_between_grid_interaction_layers_circuit(qubits=sorted(topo.nodes_as_gridqubits()), depth=4)
    shared_rt_info = cg.SharedRuntimeInfo(run_id='example', device=rainbow_device)
    rs = np.random.RandomState(10)
    placed_c, placement = hardcoded.place_circuit(circuit, problem_topology=topo, shared_rt_info=shared_rt_info, rs=rs)
    cirq.is_valid_placement(rainbow_graph, topo.graph, placement)
    assert isinstance(placed_c, cirq.FrozenCircuit)