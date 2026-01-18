from bs4 import BeautifulSoup, NavigableString, Comment, Doctype
from textwrap import fill
import re
import six
def convert_blockquote(self, el, text, convert_as_inline):
    if convert_as_inline:
        return text
    return '\n' + (line_beginning_re.sub('> ', text) + '\n\n') if text else ''