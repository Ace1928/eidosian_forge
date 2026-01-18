from bs4 import BeautifulSoup, NavigableString, Comment, Doctype
from textwrap import fill
import re
import six
def convert_a(self, el, text, convert_as_inline):
    prefix, suffix, text = chomp(text)
    if not text:
        return ''
    href = el.get('href')
    title = el.get('title')
    if self.options['autolinks'] and text.replace('\\_', '_') == href and (not title) and (not self.options['default_title']):
        return '<%s>' % href
    if self.options['default_title'] and (not title):
        title = href
    title_part = ' "%s"' % title.replace('"', '\\"') if title else ''
    return '%s[%s](%s%s)%s' % (prefix, text, href, title_part, suffix) if href else text