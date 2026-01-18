import os
import sys
from glob import glob
from warnings import warn
from distutils.core import Command
from distutils import dir_util
from distutils import file_util
from distutils import archive_util
from distutils.text_file import TextFile
from distutils.filelist import FileList
from distutils import log
from distutils.util import convert_path
from distutils.errors import DistutilsTemplateError, DistutilsOptionError
def read_template(self):
    """Read and parse manifest template file named by self.template.

        (usually "MANIFEST.in") The parsing and processing is done by
        'self.filelist', which updates itself accordingly.
        """
    log.info("reading manifest template '%s'", self.template)
    template = TextFile(self.template, strip_comments=1, skip_blanks=1, join_lines=1, lstrip_ws=1, rstrip_ws=1, collapse_join=1)
    try:
        while True:
            line = template.readline()
            if line is None:
                break
            try:
                self.filelist.process_template_line(line)
            except (DistutilsTemplateError, ValueError) as msg:
                self.warn('%s, line %d: %s' % (template.filename, template.current_line, msg))
    finally:
        template.close()