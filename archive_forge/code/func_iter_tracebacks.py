import re
import sys
import os.path as op
from glob import glob
from traits.trait_errors import TraitError
from nipype.utils.filemanip import loadcrash
def iter_tracebacks(logdir):
    """Return an iterator over each file path and
    traceback field inside `logdir`.
    Parameters
    ----------
    logdir: str
        Path to the log folder.

    field: str
        Field name to be read from the crash file.

    Yields
    ------
    path_file: str

    traceback: str
    """
    crash_files = sorted(glob(op.join(logdir, '*.pkl*')))
    for cf in crash_files:
        yield (cf, load_pklz_traceback(cf))