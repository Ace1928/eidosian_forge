from __future__ import unicode_literals
import re
from pybtex.richtext import Symbol, Text
from pybtex.style.formatting import BaseStyle, toplevel
from pybtex.style.template import (
def get_incollection_template(self, e):
    template = toplevel[sentence[self.format_names('author')], self.format_title(e, 'title'), words['In', sentence[optional[self.format_editor(e, as_sentence=False)], self.format_btitle(e, 'booktitle', as_sentence=False), self.format_volume_and_series(e, as_sentence=False), self.format_chapter_and_pages(e)]], sentence[optional_field('publisher'), optional_field('address'), self.format_edition(e), date], self.format_web_refs(e)]
    return template