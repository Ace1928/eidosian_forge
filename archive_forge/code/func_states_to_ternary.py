import itertools
import numpy as np
import pennylane as qml
from pennylane import QubitDevice
from pennylane.measurements import MeasurementProcess
from pennylane.wires import Wires
@staticmethod
def states_to_ternary(samples, num_wires, dtype=np.int64):
    """Convert basis states from base 10 to ternary representation.

        This is an auxiliary method to the generate_samples method.

        Args:
            samples (array[int]): samples of basis states in base 10 representation
            num_wires (int): the number of qutrits
            dtype (type): Type of the internal integer array to be used. Can be
                important to specify for large systems for memory allocation
                purposes.

        Returns:
            array[int]: basis states in ternary representation
        """
    ternary_arr = []
    for sample in samples:
        num = []
        for _ in range(num_wires):
            sample, r = divmod(sample, 3)
            num.append(r)
        ternary_arr.append(num[::-1])
    return np.array(ternary_arr, dtype=dtype)