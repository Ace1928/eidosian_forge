import cirq
import pytest
def test_circuit_with_tagged_ops():
    device = cirq.testing.construct_ring_device(4)
    device_graph = device.metadata.nx_graph
    q = cirq.LineQubit.range(3)
    circuit = cirq.Circuit([cirq.Moment(cirq.CNOT(q[0], q[1]).with_tags('u')), cirq.Moment(cirq.CNOT(q[1], q[2])), cirq.Moment(cirq.CNOT(q[0], q[2]).with_tags('u')), cirq.Moment(cirq.X(q[0]).with_tags('u')), cirq.Moment(cirq.X(q[0]).with_tags('u'))])
    hard_coded_mapper = cirq.HardCodedInitialMapper({q[i]: q[i] for i in range(3)})
    router = cirq.RouteCQC(device_graph)
    routed_circuit = router(circuit, initial_mapper=hard_coded_mapper)
    expected = cirq.Circuit([cirq.Moment(cirq.TaggedOperation(cirq.CNOT(q[0], q[1]), 'u')), cirq.Moment(cirq.CNOT(q[1], q[2])), cirq.Moment(cirq.SWAP(q[0], q[1])), cirq.Moment(cirq.TaggedOperation(cirq.CNOT(q[1], q[2]), 'u')), cirq.Moment(cirq.TaggedOperation(cirq.X(q[1]), 'u')), cirq.Moment(cirq.TaggedOperation(cirq.X(q[1]), 'u'))])
    cirq.testing.assert_same_circuits(routed_circuit, expected)