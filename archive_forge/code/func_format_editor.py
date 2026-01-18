from __future__ import unicode_literals
import re
from pybtex.richtext import Symbol, Text
from pybtex.style.formatting import BaseStyle, toplevel
from pybtex.style.template import (
def format_editor(self, e, as_sentence=True):
    editors = self.format_names('editor', as_sentence=False)
    if 'editor' not in e.persons:
        return editors
    if len(e.persons['editor']) > 1:
        word = 'editors'
    else:
        word = 'editor'
    result = join(sep=', ')[editors, word]
    if as_sentence:
        return sentence[result]
    else:
        return result