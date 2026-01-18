import os
import sys
from pygments.formatter import Formatter
from pygments.util import get_bool_opt, get_int_opt, get_list_opt, \
import subprocess
def get_font(self, bold, oblique):
    """
        Get the font based on bold and italic flags.
        """
    if bold and oblique:
        return self.fonts['BOLDITALIC']
    elif bold:
        return self.fonts['BOLD']
    elif oblique:
        return self.fonts['ITALIC']
    else:
        return self.fonts['NORMAL']