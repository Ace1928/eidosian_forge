import functools
import re
import shlex
import sys
from pathlib import Path
from IPython.core.magic import Magics, magics_class, line_magic
@line_magic
@is_conda_environment
def conda(self, line):
    """Run the conda package manager within the current kernel.

        Usage:
          %conda install [pkgs]
        """
    conda = _get_conda_like_executable('conda')
    self._run_command(conda, line)