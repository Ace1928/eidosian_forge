from __future__ import annotations
import os
from pymatgen.io.vasp import Potcar
def proc_dir(dirname, proc_file_function):
    """Process a directory.

    Args:
        dirname (str): Directory name.
        proc_file_function (callable): Callable to execute on directory.
    """
    for file in os.listdir(dirname):
        if os.path.isdir(os.path.join(dirname, file)):
            proc_dir(os.path.join(dirname, file), proc_file_function)
        else:
            proc_file_function(dirname, file)