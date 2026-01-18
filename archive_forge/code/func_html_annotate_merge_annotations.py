import difflib
from lxml import etree
from lxml.html import fragment_fromstring
import re
def html_annotate_merge_annotations(tokens_old, tokens_new):
    """Merge the annotations from tokens_old into tokens_new, when the
    tokens in the new document already existed in the old document.
    """
    s = InsensitiveSequenceMatcher(a=tokens_old, b=tokens_new)
    commands = s.get_opcodes()
    for command, i1, i2, j1, j2 in commands:
        if command == 'equal':
            eq_old = tokens_old[i1:i2]
            eq_new = tokens_new[j1:j2]
            copy_annotations(eq_old, eq_new)