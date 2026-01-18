import numpy as np
from qiskit.exceptions import QiskitError
def format_counts_memory(shot_memory, header=None):
    """
    Format a single bitstring (memory) from a single shot experiment.

    - The hexadecimals are expanded to bitstrings

    - Spaces are inserted at register divisions.

    Args:
        shot_memory (str): result of a single experiment.
        header (dict): the experiment header dictionary containing
            useful information for postprocessing. creg_sizes
            are a nested list where the inner element is a list
            of creg name, creg size pairs. memory_slots is an integers
            specifying the number of total memory_slots in the experiment.

    Returns:
        dict: a formatted memory
    """
    if shot_memory.startswith('0x'):
        shot_memory = _hex_to_bin(shot_memory)
    if header:
        creg_sizes = header.get('creg_sizes', None)
        memory_slots = header.get('memory_slots', None)
        if memory_slots:
            shot_memory = _pad_zeros(shot_memory, memory_slots)
        if creg_sizes and memory_slots:
            shot_memory = _separate_bitstring(shot_memory, creg_sizes)
    return shot_memory