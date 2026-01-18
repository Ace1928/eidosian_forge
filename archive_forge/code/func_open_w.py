from a disk file or from an open file, and similar for its output.
import re
import os
import tempfile
import warnings
from shlex import quote
def open_w(self, file):
    if not self.steps:
        return open(file, 'w')
    if self.steps[0][1] == SOURCE:
        raise ValueError('Template.open_w: pipeline begins with SOURCE')
    cmd = self.makepipeline('', file)
    return os.popen(cmd, 'w')