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
def make_dvi(cls, tex, fontsize):
    """
        Generate a dvi file containing latex's layout of tex string.

        Return the file name.
        """
    basefile = cls.get_basefile(tex, fontsize)
    dvifile = '%s.dvi' % basefile
    if not os.path.exists(dvifile):
        texfile = Path(cls.make_tex(tex, fontsize))
        cwd = Path(dvifile).parent
        with TemporaryDirectory(dir=cwd) as tmpdir:
            tmppath = Path(tmpdir)
            cls._run_checked_subprocess(['latex', '-interaction=nonstopmode', '--halt-on-error', f'--output-directory={tmppath.name}', f'{texfile.name}'], tex, cwd=cwd)
            (tmppath / Path(dvifile).name).replace(dvifile)
    return dvifile