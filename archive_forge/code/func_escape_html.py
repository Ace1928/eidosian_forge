from __future__ import print_function
import os
import sys
import os.path
from pygments.formatter import Formatter
from pygments.token import Token, Text, STANDARD_TYPES
from pygments.util import get_bool_opt, get_int_opt, get_list_opt, \
def escape_html(text, table=_escape_html_table):
    """Escape &, <, > as well as single and double quotes for HTML."""
    return text.translate(table)