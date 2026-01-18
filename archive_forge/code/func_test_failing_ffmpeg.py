import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import weakref
import numpy as np
import pytest
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.testing.decorators import check_figures_equal
@pytest.mark.skipif(shutil.which('/bin/sh') is None, reason='requires a POSIX OS')
def test_failing_ffmpeg(tmpdir, monkeypatch, anim):
    """
    Test that we correctly raise a CalledProcessError when ffmpeg fails.

    To do so, mock ffmpeg using a simple executable shell script that
    succeeds when called with no arguments (so that it gets registered by
    `isAvailable`), but fails otherwise, and add it to the $PATH.
    """
    with tmpdir.as_cwd():
        monkeypatch.setenv('PATH', '.:' + os.environ['PATH'])
        exe_path = Path(str(tmpdir), 'ffmpeg')
        exe_path.write_bytes(b'#!/bin/sh\n[[ $@ -eq 0 ]]\n')
        os.chmod(exe_path, 493)
        with pytest.raises(subprocess.CalledProcessError):
            anim.save('test.mpeg')