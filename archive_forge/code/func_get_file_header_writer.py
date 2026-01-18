import re
from io import BytesIO
from .lazy_import import lazy_import
from fnmatch import fnmatch
from breezy._termcolor import color_string, FG
from breezy import (
from . import controldir, errors, osutils
from . import revision as _mod_revision
from . import trace
from .revisionspec import RevisionSpec, RevisionSpec_revid, RevisionSpec_revno
def get_file_header_writer(self):
    """Get function for writing file headers"""
    write = self.outf.write
    eol_marker = self.opts.eol_marker

    def _line_writer(line):
        write(line + eol_marker)

    def _line_writer_color(line):
        write(FG.BOLD_MAGENTA + line + FG.NONE + eol_marker)
    if self.opts.show_color:
        return _line_writer_color
    else:
        return _line_writer
    return _line_writer