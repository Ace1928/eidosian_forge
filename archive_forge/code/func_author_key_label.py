from __future__ import absolute_import, unicode_literals
import re
import sys
import unicodedata
import six
from pybtex.style.labels import BaseLabelStyle
from pybtex.textutils import abbreviate
def author_key_label(self, entry):
    if not 'author' in entry.persons:
        if not 'key' in entry.fields:
            return entry.key[:3]
        else:
            return entry.fields['key'][:3]
    else:
        return self.format_lab_names(entry.persons['author'])