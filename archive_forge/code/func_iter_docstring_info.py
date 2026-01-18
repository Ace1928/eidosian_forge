import os
import re
from os.path import sep
from os.path import join as slash  # just like that name better
from os.path import dirname, abspath
import kivy
from kivy.logger import Logger
import textwrap
def iter_docstring_info(dir_name):
    """ Iterate over screenshots in directory, yield info from the file
     name and initial parse of the docstring. Errors are logged, but
     files with errors are skipped.
    """
    for file_info in iter_filename_info(dir_name):
        if 'error' in file_info:
            Logger.error(file_info['error'])
            continue
        source = slash(examples_dir, file_info['dir'], file_info['file'] + '.' + file_info['ext'])
        if not os.path.exists(source):
            Logger.error('Screen shot references source code that does not exist:  %s', source)
            continue
        with open(source) as f:
            text = f.read()
            docstring_info = parse_docstring_info(text)
            if 'error' in docstring_info:
                Logger.error(docstring_info['error'] + '  File: ' + source)
                continue
            else:
                file_info.update(docstring_info)
        yield file_info