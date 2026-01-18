from itertools import groupby
import numpy as np
from qiskit.transpiler.exceptions import TranspilerError
from qiskit.circuit.library.standard_gates.p import PhaseGate
from qiskit.circuit.library.standard_gates.u import UGate
from qiskit.circuit.library.standard_gates.u1 import U1Gate
from qiskit.circuit.library.standard_gates.u2 import U2Gate
from qiskit.circuit.library.standard_gates.u3 import U3Gate
from qiskit.circuit import ParameterExpression
from qiskit.circuit.gate import Gate
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.quantum_info.quaternion import Quaternion
from qiskit._accelerate.optimize_1q_gates import compose_u3_rust
@staticmethod
def compose_u3(theta1, phi1, lambda1, theta2, phi2, lambda2):
    """Return a triple theta, phi, lambda for the product.

        u3(theta, phi, lambda)
           = u3(theta1, phi1, lambda1).u3(theta2, phi2, lambda2)
           = Rz(phi1).Ry(theta1).Rz(lambda1+phi2).Ry(theta2).Rz(lambda2)
           = Rz(phi1).Rz(phi').Ry(theta').Rz(lambda').Rz(lambda2)
           = u3(theta', phi1 + phi', lambda2 + lambda')

        Return theta, phi, lambda.
        """
    theta, phi, lamb = compose_u3_rust(theta1, phi1, lambda1, theta2, phi2, lambda2)
    return (theta, phi, lamb)