import functools
import hashlib
import logging
import os
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook, dviread
@classmethod
def get_basefile(cls, tex, fontsize, dpi=None):
    """
        Return a filename based on a hash of the string, fontsize, and dpi.
        """
    src = cls._get_tex_source(tex, fontsize) + str(dpi)
    filehash = hashlib.md5(src.encode('utf-8')).hexdigest()
    filepath = Path(cls._texcache)
    num_letters, num_levels = (2, 2)
    for i in range(0, num_letters * num_levels, num_letters):
        filepath = filepath / Path(filehash[i:i + 2])
    filepath.mkdir(parents=True, exist_ok=True)
    return os.path.join(filepath, filehash)