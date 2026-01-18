from keyword import iskeyword
import re
from .autocall import IPyAutocall
from traitlets.config.configurable import Configurable
from .inputtransformer2 import (
from .macro import Macro
from .splitinput import LineInfo
from traitlets import (
def prefilter_lines(self, lines, continue_prompt=False):
    """Prefilter multiple input lines of text.

        This is the main entry point for prefiltering multiple lines of
        input.  This simply calls :meth:`prefilter_line` for each line of
        input.

        This covers cases where there are multiple lines in the user entry,
        which is the case when the user goes back to a multiline history
        entry and presses enter.
        """
    llines = lines.rstrip('\n').split('\n')
    if len(llines) > 1:
        out = '\n'.join([self.prefilter_line(line, lnum > 0) for lnum, line in enumerate(llines)])
    else:
        out = self.prefilter_line(llines[0], continue_prompt)
    return out