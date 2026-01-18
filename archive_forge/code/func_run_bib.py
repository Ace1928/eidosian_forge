from __future__ import annotations
import os
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from traitlets import Bool, Instance, Integer, List, Unicode, default
from nbconvert.utils import _contextlib_chdir
from .latex import LatexExporter
def run_bib(self, filename, raise_on_failure=False):
    """Run bibtex one time."""
    filename = os.path.splitext(filename)[0]

    def log_error(command, out):
        self.log.warning('%s had problems, most likely because there were no citations', command[0])
        self.log.debug('%s output: %s\n%s', command[0], command, out)
    return self.run_command(self.bib_command, filename, 1, log_error, raise_on_failure)