import filecmp
import os
from pathlib import Path
import shutil
import sys
from matplotlib.testing import subprocess_run_for_testing
import pytest
def build_sphinx_html(source_dir, doctree_dir, html_dir, extra_args=None):
    extra_args = [] if extra_args is None else extra_args
    cmd = [sys.executable, '-msphinx', '-W', '-b', 'html', '-d', str(doctree_dir), str(source_dir), str(html_dir), *extra_args]
    proc = subprocess_run_for_testing(cmd, capture_output=True, text=True, env={**os.environ, 'MPLBACKEND': ''})
    out = proc.stdout
    err = proc.stderr
    assert proc.returncode == 0, f'sphinx build failed with stdout:\n{out}\nstderr:\n{err}\n'
    if err:
        pytest.fail(f'sphinx build emitted the following warnings:\n{err}')
    assert html_dir.is_dir()