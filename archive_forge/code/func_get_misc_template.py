from __future__ import unicode_literals
import re
from pybtex.richtext import Symbol, Text
from pybtex.style.formatting import BaseStyle, toplevel
from pybtex.style.template import (
def get_misc_template(self, e):
    template = toplevel[optional[sentence[self.format_names('author')]], optional[self.format_title(e, 'title')], sentence[optional[field('howpublished')], optional[date]], sentence[optional_field('note')], self.format_web_refs(e)]
    return template