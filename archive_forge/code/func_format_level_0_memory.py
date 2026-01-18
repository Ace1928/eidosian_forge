import numpy as np
from qiskit.exceptions import QiskitError
def format_level_0_memory(memory):
    """Format an experiment result memory object for measurement level 0.

    Args:
        memory (list): Memory from experiment with `meas_level==1`. `avg` or
            `single` will be inferred from shape of result memory.

    Returns:
        np.ndarray: Measurement level 0 complex numpy array

    Raises:
        QiskitError: If the returned numpy array does not have 2 (avg) or 3 (single)
            indices.
    """
    formatted_memory = _list_to_complex_array(memory)
    if not 2 <= len(formatted_memory.shape) <= 3:
        raise QiskitError('Level zero memory is not of correct shape.')
    return formatted_memory