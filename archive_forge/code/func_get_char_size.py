import os
import sys
from pygments.formatter import Formatter
from pygments.util import get_bool_opt, get_int_opt, get_list_opt, \
import subprocess
def get_char_size(self):
    """
        Get the character size.
        """
    return self.fonts['NORMAL'].getsize('M')