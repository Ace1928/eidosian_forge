from bs4 import BeautifulSoup, NavigableString, Comment, Doctype
from textwrap import fill
import re
import six
def convert_list(self, el, text, convert_as_inline):
    nested = False
    before_paragraph = False
    if el.next_sibling and el.next_sibling.name not in ['ul', 'ol']:
        before_paragraph = True
    while el:
        if el.name == 'li':
            nested = True
            break
        el = el.parent
    if nested:
        return '\n' + self.indent(text, 1).rstrip()
    return text + ('\n' if before_paragraph else '')