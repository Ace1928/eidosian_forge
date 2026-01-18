import os
import platform
import pprint
import sys
import subprocess
from pathlib import Path
from IPython.core import release
from IPython.utils import _sysinfo, encoding
def pkg_info(pkg_path: str) -> dict:
    """Return dict describing the context of this package

    Parameters
    ----------
    pkg_path : str
        path containing __init__.py for package

    Returns
    -------
    context : dict
        with named parameters of interest
    """
    src, hsh = pkg_commit_hash(pkg_path)
    return dict(ipython_version=release.version, ipython_path=pkg_path, commit_source=src, commit_hash=hsh, sys_version=sys.version, sys_executable=sys.executable, sys_platform=sys.platform, platform=platform.platform(), os_name=os.name, default_encoding=encoding.DEFAULT_ENCODING)