from __future__ import annotations
from sympy.core.mul import Mul
from sympy.external import import_module
from sympy.physics.quantum.gate import Gate, OneQubitGate, CGate, CGateS
def circuit_plot(c, nqubits, **kwargs):
    """Draw the circuit diagram for the circuit with nqubits.

    Parameters
    ==========

    c : circuit
        The circuit to plot. Should be a product of Gate instances.
    nqubits : int
        The number of qubits to include in the circuit. Must be at least
        as big as the largest ``min_qubits`` of the gates.
    """
    return CircuitPlot(c, nqubits, **kwargs)