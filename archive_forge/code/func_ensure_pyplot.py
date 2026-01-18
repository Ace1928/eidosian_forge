import atexit
import errno
import os
import pathlib
import re
import sys
import tempfile
import ast
import warnings
import shutil
from io import StringIO
from docutils.parsers.rst import directives
from docutils.parsers.rst import Directive
from sphinx.util import logging
from traitlets.config import Config
from IPython import InteractiveShell
from IPython.core.profiledir import ProfileDir
def ensure_pyplot(self):
    """
        Ensures that pyplot has been imported into the embedded IPython shell.

        Also, makes sure to set the backend appropriately if not set already.

        """
    if not self._pyplot_imported:
        if 'matplotlib.backends' not in sys.modules:
            import matplotlib
            matplotlib.use('agg')
        self.process_input_line('import matplotlib.pyplot as plt', store_history=False)
        self._pyplot_imported = True