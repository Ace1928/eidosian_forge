import math
from typing import List
import numpy as np
import rustworkx as rx
from rustworkx.visualization import graphviz_draw
from qiskit.exceptions import QiskitError
from qiskit.utils import optionals as _optionals
from qiskit.providers.exceptions import BackendPropertyError
from qiskit.transpiler.coupling import CouplingMap
from .exceptions import VisualizationError
def plot_circuit_layout(circuit, backend, view='virtual', qubit_coordinates=None):
    """Plot the layout of a circuit transpiled for a given
    target backend.

    Args:
        circuit (QuantumCircuit): Input quantum circuit.
        backend (Backend): Target backend.
        view (str): Layout view: either 'virtual' or 'physical'.
        qubit_coordinates (Sequence): An optional sequence input (list or array being the
            most common) of 2d coordinates for each qubit. The length of the
            sequence must match the number of qubits on the backend. The sequence
            should be the planar coordinates in a 0-based square grid where each
            qubit is located.

    Returns:
        Figure: A matplotlib figure showing layout.

    Raises:
        QiskitError: Invalid view type given.
        VisualizationError: Circuit has no layout attribute.

    Example:
        .. plot::
           :include-source:

            from qiskit import QuantumCircuit, transpile
            from qiskit.providers.fake_provider import GenericBackendV2
            from qiskit.visualization import plot_circuit_layout

            ghz = QuantumCircuit(3, 3)
            ghz.h(0)
            for idx in range(1,3):
                ghz.cx(0,idx)
            ghz.measure(range(3), range(3))

            backend = GenericBackendV2(num_qubits=5)
            new_circ_lv3 = transpile(ghz, backend=backend, optimization_level=3)
            plot_circuit_layout(new_circ_lv3, backend)
    """
    if circuit._layout is None:
        raise QiskitError('Circuit has no layout. Perhaps it has not been transpiled.')
    backend_version = _get_backend_interface_version(backend)
    if backend_version <= 1:
        num_qubits = backend.configuration().n_qubits
        cmap = backend.configuration().coupling_map
        cmap_len = len(cmap)
    else:
        num_qubits = backend.num_qubits
        cmap = backend.coupling_map
        cmap_len = cmap.graph.num_edges()
    qubits = []
    qubit_labels = [''] * num_qubits
    bit_locations = {bit: {'register': register, 'index': index} for register in circuit._layout.initial_layout.get_registers() for index, bit in enumerate(register)}
    for index, qubit in enumerate(circuit._layout.initial_layout.get_virtual_bits()):
        if qubit not in bit_locations:
            bit_locations[qubit] = {'register': None, 'index': index}
    if view == 'virtual':
        for key, val in circuit._layout.initial_layout.get_virtual_bits().items():
            bit_register = bit_locations[key]['register']
            if bit_register is None or bit_register.name != 'ancilla':
                qubits.append(val)
                qubit_labels[val] = str(bit_locations[key]['index'])
    elif view == 'physical':
        for key, val in circuit._layout.initial_layout.get_physical_bits().items():
            bit_register = bit_locations[val]['register']
            if bit_register is None or bit_register.name != 'ancilla':
                qubits.append(key)
                qubit_labels[key] = str(key)
    else:
        raise VisualizationError("Layout view must be 'virtual' or 'physical'.")
    qcolors = ['#648fff'] * num_qubits
    for k in qubits:
        qcolors[k] = 'black'
    lcolors = ['#648fff'] * cmap_len
    for idx, edge in enumerate(cmap):
        if edge[0] in qubits and edge[1] in qubits:
            lcolors[idx] = 'black'
    fig = plot_gate_map(backend, qubit_color=qcolors, qubit_labels=qubit_labels, line_color=lcolors, qubit_coordinates=qubit_coordinates)
    return fig