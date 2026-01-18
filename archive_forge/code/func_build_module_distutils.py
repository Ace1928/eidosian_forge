import glob
import os
import sys
import subprocess
import tempfile
import shutil
import atexit
import textwrap
import re
import pytest
import contextlib
import numpy
from pathlib import Path
from numpy.compat import asstr
from numpy._utils import asunicode
from numpy.testing import temppath, IS_WASM
from importlib import import_module
import os
import sys
@_memoize
def build_module_distutils(source_files, config_code, module_name, **kw):
    """
    Build a module via distutils and import it.

    """
    d = get_module_dir()
    dst_sources = []
    for fn in source_files:
        if not os.path.isfile(fn):
            raise RuntimeError('%s is not a file' % fn)
        dst = os.path.join(d, os.path.basename(fn))
        shutil.copyfile(fn, dst)
        dst_sources.append(dst)
    config_code = textwrap.dedent(config_code).replace('\n', '\n    ')
    code = f"""\nimport os\nimport sys\nsys.path = {repr(sys.path)}\n\ndef configuration(parent_name='',top_path=None):\n    from numpy.distutils.misc_util import Configuration\n    config = Configuration('', parent_name, top_path)\n    {config_code}\n    return config\n\nif __name__ == "__main__":\n    from numpy.distutils.core import setup\n    setup(configuration=configuration)\n    """
    script = os.path.join(d, get_temp_module_name() + '.py')
    dst_sources.append(script)
    with open(script, 'wb') as f:
        f.write(code.encode('latin1'))
    cwd = os.getcwd()
    try:
        os.chdir(d)
        cmd = [sys.executable, script, 'build_ext', '-i']
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        out, err = p.communicate()
        if p.returncode != 0:
            raise RuntimeError('Running distutils build failed: %s\n%s' % (cmd[4:], asstr(out)))
    finally:
        os.chdir(cwd)
        for fn in dst_sources:
            os.unlink(fn)
    __import__(module_name)
    return sys.modules[module_name]