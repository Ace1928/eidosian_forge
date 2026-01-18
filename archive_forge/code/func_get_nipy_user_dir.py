import os
from os.path import join as pjoin
def get_nipy_user_dir():
    """Get the NIPY user directory

    This uses the logic in `get_home_dir` to find the home directory
    and the adds either .nipy or _nipy to the end of the path.

    We check first in environment variable ``NIPY_USER_DIR``, otherwise
    returning the default of ``<homedir>/.nipy`` (Unix) or
    ``<homedir>/_nipy`` (Windows)

    The path may well not exist; code using this routine should not
    expect the directory to exist.

    Parameters
    ----------
    None

    Returns
    -------
    nipy_dir : string
       path to user's NIPY configuration directory

    Examples
    --------
    >>> pth = get_nipy_user_dir()

    """
    try:
        return os.path.abspath(os.environ['NIPY_USER_DIR'])
    except KeyError:
        pass
    home_dir = get_home_dir()
    if os.name == 'posix':
        sdir = '.nipy'
    else:
        sdir = '_nipy'
    return pjoin(home_dir, sdir)