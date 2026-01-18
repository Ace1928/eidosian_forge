import os
import os.path as op
from warnings import warn
import numpy as np
from nibabel import load
from ... import LooseVersion
from ...utils.filemanip import split_filename
from ..base import (
from .base import FSLCommand, FSLCommandInputSpec, Info
@classmethod
def intensitymap_file_basename(cls, f):
    """Removes valid intensitymap extensions from `f`, returning a basename
        that can refer to both intensitymap files.
        """
    for ext in list(Info.ftypes.values()) + ['.txt']:
        if f.endswith(ext):
            return f[:-len(ext)]
    return f