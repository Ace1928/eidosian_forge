from __future__ import unicode_literals
import re
from pybtex.richtext import Symbol, Text
from pybtex.style.formatting import BaseStyle, toplevel
from pybtex.style.template import (
def format_author_or_editor(self, e):
    return first_of[optional[self.format_names('author')], self.format_editor(e)]