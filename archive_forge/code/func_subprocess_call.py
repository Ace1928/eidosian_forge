import os
import subprocess as sp
import sys
import warnings
import proglog
from .compat import DEVNULL
def subprocess_call(cmd, logger='bar', errorprint=True):
    """ Executes the given subprocess command.
    
    Set logger to None or a custom Proglog logger to avoid printings.
    """
    logger = proglog.default_bar_logger(logger)
    logger(message='Moviepy - Running:\n>>> "+ " ".join(cmd)')
    popen_params = {'stdout': DEVNULL, 'stderr': sp.PIPE, 'stdin': DEVNULL}
    if os.name == 'nt':
        popen_params['creationflags'] = 134217728
    proc = sp.Popen(cmd, **popen_params)
    out, err = proc.communicate()
    proc.stderr.close()
    if proc.returncode:
        if errorprint:
            logger(message='Moviepy - Command returned an error')
        raise IOError(err.decode('utf8'))
    else:
        logger(message='Moviepy - Command successful')
    del proc