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
def process_block(self, block):
    """
        process block from the block_parser and return a list of processed lines
        """
    ret = []
    output = None
    input_lines = None
    lineno = self.IP.execution_count
    input_prompt = self.promptin % lineno
    output_prompt = self.promptout % lineno
    image_file = None
    image_directive = None
    found_input = False
    for token, data in block:
        if token == COMMENT:
            out_data = self.process_comment(data)
        elif token == INPUT:
            found_input = True
            out_data, input_lines, output, is_doctest, decorator, image_file, image_directive = self.process_input(data, input_prompt, lineno)
        elif token == OUTPUT:
            if not found_input:
                TAB = ' ' * 4
                linenumber = 0
                source = 'Unavailable'
                content = 'Unavailable'
                if self.directive:
                    linenumber = self.directive.state.document.current_line
                    source = self.directive.state.document.current_source
                    content = self.directive.content
                    content = '\n'.join([TAB + line for line in content])
                e = '\n\nInvalid block: Block contains an output prompt without an input prompt.\n\nDocument source: {0}\n\nContent begins at line {1}: \n\n{2}\n\nProblematic block within content: \n\n{TAB}{3}\n\n'
                e = e.format(source, linenumber, content, block, TAB=TAB)
                sys.stdout.write(e)
                raise RuntimeError('An invalid block was detected.')
            out_data = self.process_output(data, output_prompt, input_lines, output, is_doctest, decorator, image_file)
            if out_data:
                assert ret[-1] == ''
                del ret[-1]
        if out_data:
            ret.extend(out_data)
    if image_file is not None:
        self.save_image(image_file)
    return (ret, image_directive)