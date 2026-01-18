from lxml import etree
import sys
import re
import doctest
def format_tag(self, el):
    attrs = []
    if isinstance(el, etree.CommentBase):
        return '<!--'
    for name, value in sorted(el.attrib.items()):
        attrs.append('%s="%s"' % (name, self.format_text(value, False)))
    if not attrs:
        return '<%s>' % el.tag
    return '<%s %s>' % (el.tag, ' '.join(attrs))