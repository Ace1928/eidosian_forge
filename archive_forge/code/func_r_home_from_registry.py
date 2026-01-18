import argparse
import enum
import logging
import os
import shlex
import subprocess
import sys
from typing import Optional
import warnings
def r_home_from_registry() -> Optional[str]:
    """Return the R home directory from the Windows Registry."""
    from packaging.version import Version
    try:
        import winreg
    except ImportError:
        import _winreg as winreg
    for w_hkey in [winreg.HKEY_CURRENT_USER, winreg.HKEY_LOCAL_MACHINE]:
        try:
            with winreg.OpenKeyEx(w_hkey, 'Software\\R-core\\R') as hkey:

                def get_version(i):
                    try:
                        return Version(winreg.EnumKey(hkey, i))
                    except Exception:
                        return None
                latest = max((v for v in (get_version(i) for i in range(winreg.QueryInfoKey(hkey)[0])) if v is not None))
                with winreg.OpenKeyEx(hkey, f'{latest}') as subkey:
                    r_home = winreg.QueryValueEx(subkey, 'InstallPath')[0]
                if not r_home:
                    r_home = winreg.QueryValueEx(hkey, 'InstallPath')[0]
        except Exception:
            pass
        else:
            if sys.version_info[0] == 2:
                r_home = r_home.encode(sys.getfilesystemencoding())
            break
    else:
        logger.error('Unable to determine R home.')
        r_home = None
    return r_home