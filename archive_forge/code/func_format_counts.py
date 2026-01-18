import numpy as np
from qiskit.exceptions import QiskitError
def format_counts(counts, header=None):
    """Format a single experiment result coming from backend to present
    to the Qiskit user.

    Args:
        counts (dict): counts histogram of multiple shots
        header (dict): the experiment header dictionary containing
            useful information for postprocessing.

    Returns:
        dict: a formatted counts
    """
    counts_dict = {}
    for key, val in counts.items():
        key = format_counts_memory(key, header)
        counts_dict[key] = val
    return counts_dict