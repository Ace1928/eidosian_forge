from __future__ import unicode_literals
import re
from pybtex.richtext import Symbol, Text
from pybtex.style.formatting import BaseStyle, toplevel
from pybtex.style.template import (
def get_booklet_template(self, e):
    template = toplevel[self.format_names('author'), self.format_title(e, 'title'), sentence[optional_field('howpublished'), optional_field('address'), date, optional_field('note')], self.format_web_refs(e)]
    return template