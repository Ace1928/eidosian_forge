import warnings
from warnings import warn
import breezy
def deprecated_in(version_tuple):
    """Generate a message that something was deprecated in a release.

    >>> deprecated_in((1, 4, 0))
    '%s was deprecated in version 1.4.0.'
    """
    return '%%s was deprecated in version %s.' % breezy._format_version_tuple(version_tuple)