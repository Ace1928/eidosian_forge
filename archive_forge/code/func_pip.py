import functools
import re
import shlex
import sys
from pathlib import Path
from IPython.core.magic import Magics, magics_class, line_magic
@line_magic
def pip(self, line):
    """Run the pip package manager within the current kernel.

        Usage:
          %pip install [pkgs]
        """
    python = sys.executable
    if sys.platform == 'win32':
        python = '"' + python + '"'
    else:
        python = shlex.quote(python)
    self.shell.system(' '.join([python, '-m', 'pip', line]))
    print('Note: you may need to restart the kernel to use updated packages.')