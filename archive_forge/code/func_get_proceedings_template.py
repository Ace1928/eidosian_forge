from __future__ import unicode_literals
import re
from pybtex.richtext import Symbol, Text
from pybtex.style.formatting import BaseStyle, toplevel
from pybtex.style.template import (
def get_proceedings_template(self, e):
    if 'editor' in e.persons:
        main_part = [self.format_editor(e), sentence[self.format_btitle(e, 'title', as_sentence=False), self.format_volume_and_series(e, as_sentence=False), self.format_address_organization_publisher_date(e)]]
    else:
        main_part = [optional[sentence[field('organization')]], sentence[self.format_btitle(e, 'title', as_sentence=False), self.format_volume_and_series(e, as_sentence=False), self.format_address_organization_publisher_date(e, include_organization=False)]]
    template = toplevel[main_part + [sentence[optional_field('note')], self.format_web_refs(e)]]
    return template