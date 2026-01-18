import os
import sys
from typing import Optional, Tuple
def libc_ver() -> Tuple[str, str]:
    """Try to determine the glibc version

    Returns a tuple of strings (lib, version) which default to empty strings
    in case the lookup fails.
    """
    glibc_version = glibc_version_string()
    if glibc_version is None:
        return ('', '')
    else:
        return ('glibc', glibc_version)