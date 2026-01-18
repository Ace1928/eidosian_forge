from __future__ import print_function
import atexit
import optparse
import os
import sys
import textwrap
import time
import unittest
import psutil
from psutil._common import hilite
from psutil._common import print_color
from psutil._common import term_supports_colors
from psutil._compat import super
from psutil.tests import CI_TESTING
from psutil.tests import import_module_by_path
from psutil.tests import print_sysinfo
from psutil.tests import reap_children
from psutil.tests import safe_rmpath
def get_runner(parallel=False):

    def warn(msg):
        cprint(msg + ' Running serial tests instead.', 'red')
    if parallel:
        if psutil.WINDOWS:
            warn("Can't run parallel tests on Windows.")
        elif concurrencytest is None:
            warn('concurrencytest module is not installed.')
        elif NWORKERS == 1:
            warn('Only 1 CPU available.')
        else:
            return ParallelRunner(verbosity=VERBOSITY)
    return ColouredTextRunner(verbosity=VERBOSITY)