from __future__ import annotations
import os
import subprocess
import json
import pathlib
import shutil
import tempfile
import locale
from .. import mlog
from .core import MesonException
from .universal import is_windows, windows_detect_native_arch
def setup_vsenv(force: bool=False) -> bool:
    try:
        return _setup_vsenv(force)
    except MesonException as e:
        if force:
            raise
        mlog.warning('Failed to activate VS environment:', str(e))
        return False