import sys
import os
from pathlib import Path
import io
def npy_load_module(name, fn, info=None):
    """
    Load a module. Uses ``load_module`` which will be deprecated in python
    3.12. An alternative that uses ``exec_module`` is in
    numpy.distutils.misc_util.exec_mod_from_location

    .. versionadded:: 1.11.2

    Parameters
    ----------
    name : str
        Full module name.
    fn : str
        Path to module file.
    info : tuple, optional
        Only here for backward compatibility with Python 2.*.

    Returns
    -------
    mod : module

    """
    from importlib.machinery import SourceFileLoader
    return SourceFileLoader(name, fn).load_module()