from bs4 import BeautifulSoup, NavigableString, Comment, Doctype
from textwrap import fill
import re
import six
def convert_li(self, el, text, convert_as_inline):
    parent = el.parent
    if parent is not None and parent.name == 'ol':
        if parent.get('start'):
            start = int(parent.get('start'))
        else:
            start = 1
        bullet = '%s.' % (start + parent.index(el))
    else:
        depth = -1
        while el:
            if el.name == 'ul':
                depth += 1
            el = el.parent
        bullets = self.options['bullets']
        bullet = bullets[depth % len(bullets)]
    return '%s %s\n' % (bullet, (text or '').strip())