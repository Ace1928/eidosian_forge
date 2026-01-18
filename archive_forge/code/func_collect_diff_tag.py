from lxml import etree
import sys
import re
import doctest
def collect_diff_tag(self, want, got):
    if not self.tag_compare(want.tag, got.tag):
        tag = '%s (got: %s)' % (want.tag, got.tag)
    else:
        tag = got.tag
    attrs = []
    any = want.tag == 'any' or 'any' in want.attrib
    for name, value in sorted(got.attrib.items()):
        if name not in want.attrib and (not any):
            attrs.append('+%s="%s"' % (name, self.format_text(value, False)))
        else:
            if name in want.attrib:
                text = self.collect_diff_text(want.attrib[name], value, False)
            else:
                text = self.format_text(value, False)
            attrs.append('%s="%s"' % (name, text))
    if not any:
        for name, value in sorted(want.attrib.items()):
            if name in got.attrib:
                continue
            attrs.append('-%s="%s"' % (name, self.format_text(value, False)))
    if attrs:
        tag = '<%s %s>' % (tag, ' '.join(attrs))
    else:
        tag = '<%s>' % tag
    return tag