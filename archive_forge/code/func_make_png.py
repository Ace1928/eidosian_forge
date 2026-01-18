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
def make_png(cls, tex, fontsize, dpi):
    """
        Generate a png file containing latex's rendering of tex string.

        Return the file name.
        """
    basefile = cls.get_basefile(tex, fontsize, dpi)
    pngfile = '%s.png' % basefile
    if not os.path.exists(pngfile):
        dvifile = cls.make_dvi(tex, fontsize)
        cmd = ['dvipng', '-bg', 'Transparent', '-D', str(dpi), '-T', 'tight', '-o', pngfile, dvifile]
        if getattr(mpl, '_called_from_pytest', False) and mpl._get_executable_info('dvipng').raw_version != '1.16':
            cmd.insert(1, '--freetype0')
        cls._run_checked_subprocess(cmd, tex)
    return pngfile