import re
import shutil
import subprocess
import warnings
from io import BytesIO, TextIOWrapper
from nbconvert.utils.version import check_version
from .exceptions import ConversionException
def check_pandoc_version():
    """Returns True if pandoc's version meets at least minimal version.

    Raises
    ------
    PandocMissing
        If pandoc is unavailable.
    """
    if check_pandoc_version._cached is not None:
        return check_pandoc_version._cached
    v = get_pandoc_version()
    if v is None:
        warnings.warn('Sorry, we cannot determine the version of pandoc.\nPlease consider reporting this issue and include theoutput of pandoc --version.\nContinuing...', RuntimeWarning, stacklevel=2)
        return False
    ok = check_version(v, _minimal_version, max_v=_maximal_version)
    check_pandoc_version._cached = ok
    if not ok:
        warnings.warn('You are using an unsupported version of pandoc (%s).\n' % v + 'Your version must be at least (%s) ' % _minimal_version + 'but less than (%s).\n' % _maximal_version + 'Refer to https://pandoc.org/installing.html.\nContinuing with doubts...', RuntimeWarning, stacklevel=2)
    return ok