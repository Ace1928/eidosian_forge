from __future__ import unicode_literals
import re
from pybtex.richtext import Symbol, Text
from pybtex.style.formatting import BaseStyle, toplevel
from pybtex.style.template import (
def format_edition(self, e):
    return optional[words[field('edition', apply_func=lambda x: x.lower()), 'edition']]