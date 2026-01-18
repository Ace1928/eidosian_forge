from __future__ import annotations
import numpy as np
def make_dict_observable(matrix_observable: list | np.ndarray) -> dict:
    """Convert an observable in matrix form to dictionary form.

    Takes in a diagonal observable as a matrix and converts it to a dictionary
    form. Can also handle a list sorted of the diagonal elements.

    Args:
        matrix_observable (list): The observable to be converted to dictionary
        form. Can be a matrix or just an ordered list of observed values

    Returns:
        Dict: A dictionary with all observable states as keys, and corresponding
        values being the observed value for that state
    """
    dict_observable = {}
    observable = np.array(matrix_observable)
    observable_size = len(observable)
    observable_bits = int(np.ceil(np.log2(observable_size)))
    binary_formatter = f'0{observable_bits}b'
    if observable.ndim == 2:
        observable = observable.diagonal()
    for state_no in range(observable_size):
        state_str = format(state_no, binary_formatter)
        dict_observable[state_str] = observable[state_no]
    return dict_observable