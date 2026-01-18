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
def process_output(self, data, output_prompt, input_lines, output, is_doctest, decorator, image_file):
    """
        Process data block for OUTPUT token.

        """
    TAB = ' ' * 4
    if is_doctest and output is not None:
        found = output
        found = found.strip()
        submitted = data.strip()
        if self.directive is None:
            source = 'Unavailable'
            content = 'Unavailable'
        else:
            source = self.directive.state.document.current_source
            content = self.directive.content
            content = '\n'.join([TAB + line for line in content])
        ind = found.find(output_prompt)
        if ind < 0:
            e = 'output does not contain output prompt\n\nDocument source: {0}\n\nRaw content: \n{1}\n\nInput line(s):\n{TAB}{2}\n\nOutput line(s):\n{TAB}{3}\n\n'
            e = e.format(source, content, '\n'.join(input_lines), repr(found), TAB=TAB)
            raise RuntimeError(e)
        found = found[len(output_prompt):].strip()
        if decorator.strip() == '@doctest':
            if found != submitted:
                e = 'doctest failure\n\nDocument source: {0}\n\nRaw content: \n{1}\n\nOn input line(s):\n{TAB}{2}\n\nwe found output:\n{TAB}{3}\n\ninstead of the expected:\n{TAB}{4}\n\n'
                e = e.format(source, content, '\n'.join(input_lines), repr(found), repr(submitted), TAB=TAB)
                raise RuntimeError(e)
        else:
            self.custom_doctest(decorator, input_lines, found, submitted)
    out_data = []
    is_verbatim = decorator == '@verbatim' or self.is_verbatim
    if is_verbatim and data.strip():
        out_data.append('{0} {1}\n'.format(output_prompt, data))
    return out_data