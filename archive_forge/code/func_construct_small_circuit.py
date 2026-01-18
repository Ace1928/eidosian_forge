import networkx as nx
import pytest
import cirq
def construct_small_circuit():
    return cirq.Circuit([cirq.Moment(cirq.CNOT(cirq.NamedQubit('1'), cirq.NamedQubit('3'))), cirq.Moment(cirq.CNOT(cirq.NamedQubit('2'), cirq.NamedQubit('3'))), cirq.Moment(cirq.CNOT(cirq.NamedQubit('4'), cirq.NamedQubit('3')), cirq.X(cirq.NamedQubit('5')))])