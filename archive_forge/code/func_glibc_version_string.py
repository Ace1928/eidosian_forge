import os
import sys
from typing import Optional, Tuple
def glibc_version_string() -> Optional[str]:
    """Returns glibc version string, or None if not using glibc."""
    return glibc_version_string_confstr() or glibc_version_string_ctypes()