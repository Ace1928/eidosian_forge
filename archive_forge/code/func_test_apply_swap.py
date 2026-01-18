from networkx.utils.misc import graphs_equal
import pytest
import networkx as nx
import cirq
def test_apply_swap():
    device_graph, initial_mapping, q = construct_device_graph_and_mapping()
    mm = cirq.MappingManager(device_graph, initial_mapping)
    q_int = [mm.logical_qid_to_int[q[i]] if q[i] in initial_mapping else -1 for i in range(len(q))]
    with pytest.raises(ValueError):
        mm.apply_swap(q_int[1], q_int[2])
    logical_to_physical_before_swap = mm.logical_to_physical.copy()
    mm.apply_swap(q_int[1], q_int[1])
    assert all(logical_to_physical_before_swap == mm.logical_to_physical)
    mm.apply_swap(q_int[1], q_int[3])
    mm.apply_swap(q_int[1], q_int[3])
    assert all(logical_to_physical_before_swap == mm.logical_to_physical)
    for i in range(len(mm.logical_to_physical)):
        assert mm.logical_to_physical[mm.physical_to_logical[i]] == i
        assert mm.physical_to_logical[mm.logical_to_physical[i]] == i