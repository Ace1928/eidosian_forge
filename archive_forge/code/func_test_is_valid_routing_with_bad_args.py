import pytest
import networkx as nx
import cirq
import cirq.contrib.routing as ccr
def test_is_valid_routing_with_bad_args():
    p, q, r = cirq.LineQubit.range(3)
    x, y = (cirq.NamedQubit('x'), cirq.NamedQubit('y'))
    circuit = cirq.Circuit([cirq.CNOT(x, y), cirq.CZ(x, y)])
    routed_circuit = cirq.Circuit([cirq.CNOT(p, q), cirq.CZ(q, r)])
    initial_mapping = {p: x, q: y}
    swap_network = ccr.SwapNetwork(routed_circuit, initial_mapping)
    assert not ccr.is_valid_routing(circuit, swap_network)

    def equals(*args):
        raise ValueError
    with pytest.raises(ValueError):
        ccr.is_valid_routing(circuit, swap_network, equals=equals)