from __future__ import unicode_literals
import re
from pybtex.richtext import Symbol, Text
from pybtex.style.formatting import BaseStyle, toplevel
from pybtex.style.template import (
def get_inbook_template(self, e):
    template = toplevel[self.format_author_or_editor(e), sentence[self.format_btitle(e, 'title', as_sentence=False), self.format_chapter_and_pages(e)], self.format_volume_and_series(e), sentence[field('publisher'), optional_field('address'), optional[words[field('edition'), 'edition']], date, optional_field('note')], self.format_web_refs(e)]
    return template