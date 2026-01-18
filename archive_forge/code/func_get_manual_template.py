from __future__ import unicode_literals
import re
from pybtex.richtext import Symbol, Text
from pybtex.style.formatting import BaseStyle, toplevel
from pybtex.style.template import (
def get_manual_template(self, e):
    template = toplevel[optional[sentence[self.format_names('author')]], self.format_btitle(e, 'title'), sentence[optional_field('organization'), optional_field('address'), self.format_edition(e), optional[date]], sentence[optional_field('note')], self.format_web_refs(e)]
    return template