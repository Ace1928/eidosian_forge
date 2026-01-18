from bs4 import BeautifulSoup, NavigableString, Comment, Doctype
from textwrap import fill
import re
import six
def convert_img(self, el, text, convert_as_inline):
    alt = el.attrs.get('alt', None) or ''
    src = el.attrs.get('src', None) or ''
    title = el.attrs.get('title', None) or ''
    title_part = ' "%s"' % title.replace('"', '\\"') if title else ''
    if convert_as_inline and el.parent.name not in self.options['keep_inline_images_in']:
        return alt
    return '![%s](%s%s)' % (alt, src, title_part)