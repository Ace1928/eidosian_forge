import inspect
import llvmlite.binding as ll
import multiprocessing
import numpy as np
import os
import stat
import shutil
import subprocess
import sys
import traceback
import unittest
import warnings
from numba import njit
from numba.core import codegen
from numba.core.caching import _UserWideCacheLocator
from numba.core.errors import NumbaWarning
from numba.parfors import parfor
from numba.tests.support import (
from numba import njit
from numba import njit
from file2 import function2
from numba import njit
@unittest.skipIf(ipykernel is None or ipykernel.version_info[0] < 6, 'requires ipykernel >= 6')
def test_ipykernel(self):
    base_cmd = [sys.executable, '-m', 'IPython']
    base_cmd += ['--quiet', '--quick', '--no-banner', '--colors=NoColor']
    try:
        ver = subprocess.check_output(base_cmd + ['--version'])
    except subprocess.CalledProcessError as e:
        self.skipTest('ipython not available: return code %d' % e.returncode)
    ver = ver.strip().decode()
    from ipykernel import compiler
    inputfn = compiler.get_tmp_directory()
    with open(inputfn, 'w') as f:
        f.write('\n                import os\n                import sys\n\n                from numba import jit\n\n                # IPython 5 does not support multiline input if stdin isn\'t\n                # a tty (https://github.com/ipython/ipython/issues/9752)\n                f = jit(cache=True)(lambda: 42)\n\n                res = f()\n                # IPython writes on stdout, so use stderr instead\n                sys.stderr.write(u"cache hits = %d\\n" % f.stats.cache_hits[()])\n\n                # IPython hijacks sys.exit(), bypass it\n                sys.stdout.flush()\n                sys.stderr.flush()\n                os._exit(res)\n                ')

    def execute_with_input():
        with open(inputfn, 'rb') as stdin:
            p = subprocess.Popen(base_cmd, stdin=stdin, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
            out, err = p.communicate()
            if p.returncode != 42:
                self.fail('unexpected return code %d\n-- stdout:\n%s\n-- stderr:\n%s\n' % (p.returncode, out, err))
            return err
    execute_with_input()
    err = execute_with_input()
    self.assertIn('cache hits = 1', err.strip())