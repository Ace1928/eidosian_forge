import os
from ... import LooseVersion
from ...utils.filemanip import fname_presuffix
from ..base import (
def no_freesurfer():
    """Checks if FreeSurfer is NOT installed
    used with skipif to skip tests that will
    fail if FreeSurfer is not installed"""
    if Info.version() is None:
        return True
    else:
        return False