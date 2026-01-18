import numpy as np
from qiskit.transpiler.basepasses import TransformationPass
from qiskit.dagcircuit import DAGCircuit
from qiskit.circuit.library.standard_gates import RXGate, RZGate, SXGate, XGate
def quantize_angles(self, qubit, original_angle):
    """Quantize the RX rotation angles by assigning the same value for the angles
        that differ within a resolution provided by the user.

        Args:
            qubit (qiskit.circuit.Qubit): This will be the dict key to access the list of
                quantized rotation angles.
            original_angle (float): Original rotation angle, before quantization.

        Returns:
            float: Quantized angle.
        """
    if (angles := self.already_generated.get(qubit)) is None:
        self.already_generated[qubit] = np.array([original_angle])
        return original_angle
    similar_angles = angles[np.isclose(angles, original_angle, atol=self.resolution_in_radian / 2)]
    if similar_angles.size == 0:
        self.already_generated[qubit] = np.append(angles, original_angle)
        return original_angle
    return float(similar_angles[0])