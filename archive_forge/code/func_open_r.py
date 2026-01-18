from a disk file or from an open file, and similar for its output.
import re
import os
import tempfile
import warnings
from shlex import quote
def open_r(self, file):
    """t.open_r(file) and t.open_w(file) implement
        t.open(file, 'r') and t.open(file, 'w') respectively."""
    if not self.steps:
        return open(file, 'r')
    if self.steps[-1][1] == SINK:
        raise ValueError('Template.open_r: pipeline ends width SINK')
    cmd = self.makepipeline(file, '')
    return os.popen(cmd, 'r')