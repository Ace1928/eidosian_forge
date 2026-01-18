import os
from os.path import join as pjoin
def get_nipy_system_dir():
    """Get systemwide NIPY configuration file directory

    On posix systems this will be ``/etc/nipy``.
    On Windows, the directory is less useful, but by default it will be
    ``C:\\etc\\nipy``

    The path may well not exist; code using this routine should not
    expect the directory to exist.

    Parameters
    ----------
    None

    Returns
    -------
    nipy_dir : string
       path to systemwide NIPY configuration directory

    Examples
    --------
    >>> pth = get_nipy_system_dir()
    """
    if os.name == 'nt':
        return 'C:\\etc\\nipy'
    if os.name == 'posix':
        return '/etc/nipy'