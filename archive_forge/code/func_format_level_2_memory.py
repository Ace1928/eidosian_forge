import numpy as np
from qiskit.exceptions import QiskitError
def format_level_2_memory(memory, header=None):
    """Format an experiment result memory object for measurement level 2.

    Args:
        memory (list): Memory from experiment with `meas_level==2` and `memory==True`.
        header (dict): the experiment header dictionary containing
            useful information for postprocessing.

    Returns:
        list[str]: List of bitstrings
    """
    memory_list = []
    for shot_memory in memory:
        memory_list.append(format_counts_memory(shot_memory, header))
    return memory_list