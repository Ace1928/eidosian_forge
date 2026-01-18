from bs4 import BeautifulSoup, NavigableString, Comment, Doctype
from textwrap import fill
import re
import six
def convert_hn(self, n, el, text, convert_as_inline):
    if convert_as_inline:
        return text
    style = self.options['heading_style'].lower()
    text = text.rstrip()
    if style == UNDERLINED and n <= 2:
        line = '=' if n == 1 else '-'
        return self.underline(text, line)
    hashes = '#' * n
    if style == ATX_CLOSED:
        return '%s %s %s\n\n' % (hashes, text, hashes)
    return '%s %s\n\n' % (hashes, text)