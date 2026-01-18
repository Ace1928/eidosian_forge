import sys
import re
import os
from configparser import RawConfigParser
def read_config(pkgname, dirs=None):
    """
    Return library info for a package from its configuration file.

    Parameters
    ----------
    pkgname : str
        Name of the package (should match the name of the .ini file, without
        the extension, e.g. foo for the file foo.ini).
    dirs : sequence, optional
        If given, should be a sequence of directories - usually including
        the NumPy base directory - where to look for npy-pkg-config files.

    Returns
    -------
    pkginfo : class instance
        The `LibraryInfo` instance containing the build information.

    Raises
    ------
    PkgNotFound
        If the package is not found.

    See Also
    --------
    misc_util.get_info, misc_util.get_pkg_info

    Examples
    --------
    >>> npymath_info = np.distutils.npy_pkg_config.read_config('npymath')
    >>> type(npymath_info)
    <class 'numpy.distutils.npy_pkg_config.LibraryInfo'>
    >>> print(npymath_info)
    Name: npymath
    Description: Portable, core math library implementing C99 standard
    Requires:
    Version: 0.1  #random

    """
    try:
        return _CACHE[pkgname]
    except KeyError:
        v = _read_config_imp(pkg_to_filename(pkgname), dirs)
        _CACHE[pkgname] = v
        return v