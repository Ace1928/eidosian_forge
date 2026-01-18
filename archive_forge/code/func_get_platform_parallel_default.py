from the multiprocessing library.
import os
from concurrent.futures import ProcessPoolExecutor
import sys
from qiskit.exceptions import QiskitError
from qiskit.utils.multiprocessing import local_hardware_info
from qiskit import user_config
def get_platform_parallel_default():
    """
    Returns the default parallelism flag value for the current platform.

    Returns:
        parallel_default: The default parallelism flag value for the
        current platform.

    """
    if sys.platform == 'win32':
        parallel_default = False
    elif sys.platform == 'darwin':
        parallel_default = False
    else:
        parallel_default = True
    return parallel_default