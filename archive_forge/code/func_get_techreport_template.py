from __future__ import unicode_literals
import re
from pybtex.richtext import Symbol, Text
from pybtex.style.formatting import BaseStyle, toplevel
from pybtex.style.template import (
def get_techreport_template(self, e):
    template = toplevel[sentence[self.format_names('author')], self.format_title(e, 'title'), sentence[words[first_of[optional_field('type'), 'Technical Report'], optional_field('number')], field('institution'), optional_field('address'), date], sentence[optional_field('note')], self.format_web_refs(e)]
    return template