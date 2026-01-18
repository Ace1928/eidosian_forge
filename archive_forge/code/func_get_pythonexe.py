import os
import sys
import subprocess
import locale
import warnings
from numpy.distutils.misc_util import is_sequence, make_temp_file
from numpy.distutils import log
def get_pythonexe():
    pythonexe = sys.executable
    if os.name in ['nt', 'dos']:
        fdir, fn = os.path.split(pythonexe)
        fn = fn.upper().replace('PYTHONW', 'PYTHON')
        pythonexe = os.path.join(fdir, fn)
        assert os.path.isfile(pythonexe), '%r is not a file' % (pythonexe,)
    return pythonexe