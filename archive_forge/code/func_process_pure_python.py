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
def process_pure_python(self, content):
    """
        content is a list of strings. it is unedited directive content

        This runs it line by line in the InteractiveShell, prepends
        prompts as needed capturing stderr and stdout, then returns
        the content as a list as if it were ipython code
        """
    output = []
    savefig = False
    multiline = False
    multiline_start = None
    fmtin = self.promptin
    ct = 0
    for lineno, line in enumerate(content):
        line_stripped = line.strip()
        if not len(line):
            output.append(line)
            continue
        if any((line_stripped.startswith('@' + pseudo_decorator) for pseudo_decorator in PSEUDO_DECORATORS)):
            output.extend([line])
            if 'savefig' in line:
                savefig = True
            continue
        if line_stripped.startswith('#'):
            output.extend([line])
            continue
        continuation = u'   %s:' % ''.join(['.'] * (len(str(ct)) + 2))
        if not multiline:
            modified = u'%s %s' % (fmtin % ct, line_stripped)
            output.append(modified)
            ct += 1
            try:
                ast.parse(line_stripped)
                output.append(u'')
            except Exception:
                multiline = True
                multiline_start = lineno
        else:
            modified = u'%s %s' % (continuation, line)
            output.append(modified)
            if len(content) > lineno + 1:
                nextline = content[lineno + 1]
                if len(nextline) - len(nextline.lstrip()) > 3:
                    continue
            try:
                mod = ast.parse('\n'.join(content[multiline_start:lineno + 1]))
                if isinstance(mod.body[0], ast.FunctionDef):
                    for element in mod.body[0].body:
                        if isinstance(element, ast.Return):
                            multiline = False
                else:
                    output.append(u'')
                    multiline = False
            except Exception:
                pass
        if savefig:
            self.ensure_pyplot()
            self.process_input_line('plt.clf()', store_history=False)
            self.clear_cout()
            savefig = False
    return output