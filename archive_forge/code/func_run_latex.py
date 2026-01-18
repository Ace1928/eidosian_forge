from __future__ import annotations
import os
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from traitlets import Bool, Instance, Integer, List, Unicode, default
from nbconvert.utils import _contextlib_chdir
from .latex import LatexExporter
def run_latex(self, filename, raise_on_failure=LatexFailed):
    """Run xelatex self.latex_count times."""

    def log_error(command, out):
        self.log.critical('%s failed: %s\n%s', command[0], command, out)
    return self.run_command(self.latex_command, filename, self.latex_count, log_error, raise_on_failure)