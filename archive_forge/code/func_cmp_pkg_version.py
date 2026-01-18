from __future__ import annotations
import sys
from contextlib import suppress
from subprocess import run
from packaging.version import Version
def cmp_pkg_version(version_str: str, pkg_version_str: str=__version__) -> int:
    """Compare ``version_str`` to current package version

    This comparator follows `PEP-440`_ conventions for determining version
    ordering.

    To be valid, a version must have a numerical major version. It may be
    optionally followed by a dot and a numerical minor version, which may,
    in turn, optionally be followed by a dot and a numerical micro version,
    and / or by an "extra" string.
    The extra string may further contain a "+". Any value to the left of a "+"
    labels the version as pre-release, while values to the right indicate a
    post-release relative to the values to the left. That is,
    ``1.2.0+1`` is post-release for ``1.2.0``, while ``1.2.0rc1+1`` is
    post-release for ``1.2.0rc1`` and pre-release for ``1.2.0``.

    Parameters
    ----------
    version_str : str
        Version string to compare to current package version
    pkg_version_str : str, optional
        Version of our package.  Optional, set from ``__version__`` by default.

    Returns
    -------
    version_cmp : int
        1 if `version_str` is a later version than `pkg_version_str`, 0 if
        same, -1 if earlier.

    Examples
    --------
    >>> cmp_pkg_version('1.2.1', '1.2.0')
    1
    >>> cmp_pkg_version('1.2.0dev', '1.2.0')
    -1
    >>> cmp_pkg_version('1.2.0dev', '1.2.0rc1')
    -1
    >>> cmp_pkg_version('1.2.0rc1', '1.2.0')
    -1
    >>> cmp_pkg_version('1.2.0rc1+1', '1.2.0rc1')
    1
    >>> cmp_pkg_version('1.2.0rc1+1', '1.2.0')
    -1
    >>> cmp_pkg_version('1.2.0.post1', '1.2.0')
    1

    .. _`PEP-440`: https://www.python.org/dev/peps/pep-0440/
    """
    return _cmp(Version(version_str), Version(pkg_version_str))