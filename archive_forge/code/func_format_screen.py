import os
import re
import string
import textwrap
import warnings
from string import Formatter
from pathlib import Path
from typing import List, Dict, Tuple, Optional, cast
def format_screen(strng):
    """Format a string for screen printing.

    This removes some latex-type format codes."""
    par_re = re.compile('\\\\$', re.MULTILINE)
    strng = par_re.sub('', strng)
    return strng