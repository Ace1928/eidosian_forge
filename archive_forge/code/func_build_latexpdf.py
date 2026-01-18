import the main Sphinx modules (like sphinx.applications, sphinx.builders).
import os
import subprocess
import sys
from os import path
from typing import List, Optional
import sphinx
from sphinx.cmd.build import build_main
from sphinx.util.console import blue, bold, color_terminal, nocolor  # type: ignore
from sphinx.util.osutil import cd, rmtree
def build_latexpdf(self) -> int:
    if self.run_generic_build('latex') > 0:
        return 1
    if sys.platform == 'win32':
        makecmd = os.environ.get('MAKE', 'make.bat')
    else:
        makecmd = self.makecmd
    try:
        with cd(self.builddir_join('latex')):
            return subprocess.call([makecmd, 'all-pdf'])
    except OSError:
        print('Error: Failed to run: %s' % makecmd)
        return 1