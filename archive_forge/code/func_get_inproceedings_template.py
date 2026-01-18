from __future__ import unicode_literals
import re
from pybtex.richtext import Symbol, Text
from pybtex.style.formatting import BaseStyle, toplevel
from pybtex.style.template import (
def get_inproceedings_template(self, e):
    template = toplevel[sentence[self.format_names('author')], self.format_title(e, 'title'), words['In', sentence[optional[self.format_editor(e, as_sentence=False)], self.format_btitle(e, 'booktitle', as_sentence=False), self.format_volume_and_series(e, as_sentence=False), optional[pages]], self.format_address_organization_publisher_date(e)], sentence[optional_field('note')], self.format_web_refs(e)]
    return template