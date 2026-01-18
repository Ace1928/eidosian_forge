from pathlib import Path
from tempfile import TemporaryDirectory
import locale
import logging
import os
import subprocess
import sys
import matplotlib as mpl
from matplotlib import _api
def subprocess_run_for_testing(command, env=None, timeout=None, stdout=None, stderr=None, check=False, text=True, capture_output=False):
    """
    Create and run a subprocess.

    Thin wrapper around `subprocess.run`, intended for testing.  Will
    mark fork() failures on Cygwin as expected failures: not a
    success, but not indicating a problem with the code either.

    Parameters
    ----------
    args : list of str
    env : dict[str, str]
    timeout : float
    stdout, stderr
    check : bool
    text : bool
        Also called ``universal_newlines`` in subprocess.  I chose this
        name since the main effect is returning bytes (`False`) vs. str
        (`True`), though it also tries to normalize newlines across
        platforms.
    capture_output : bool
        Set stdout and stderr to subprocess.PIPE

    Returns
    -------
    proc : subprocess.Popen

    See Also
    --------
    subprocess.run

    Raises
    ------
    pytest.xfail
        If platform is Cygwin and subprocess reports a fork() failure.
    """
    if capture_output:
        stdout = stderr = subprocess.PIPE
    try:
        proc = subprocess.run(command, env=env, timeout=timeout, check=check, stdout=stdout, stderr=stderr, text=text)
    except BlockingIOError:
        if sys.platform == 'cygwin':
            import pytest
            pytest.xfail('Fork failure')
        raise
    return proc