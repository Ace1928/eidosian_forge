from lxml import etree
import sys
import re
import doctest
def collect_diff(self, want, got, html, indent):
    parts = []
    if not len(want) and (not len(got)):
        parts.append(' ' * indent)
        parts.append(self.collect_diff_tag(want, got))
        if not self.html_empty_tag(got, html):
            parts.append(self.collect_diff_text(want.text, got.text))
            parts.append(self.collect_diff_end_tag(want, got))
        parts.append(self.collect_diff_text(want.tail, got.tail))
        parts.append('\n')
        return ''.join(parts)
    parts.append(' ' * indent)
    parts.append(self.collect_diff_tag(want, got))
    parts.append('\n')
    if strip(want.text) or strip(got.text):
        parts.append(' ' * indent)
        parts.append(self.collect_diff_text(want.text, got.text))
        parts.append('\n')
    want_children = list(want)
    got_children = list(got)
    while want_children or got_children:
        if not want_children:
            parts.append(self.format_doc(got_children.pop(0), html, indent + 2, '+'))
            continue
        if not got_children:
            parts.append(self.format_doc(want_children.pop(0), html, indent + 2, '-'))
            continue
        parts.append(self.collect_diff(want_children.pop(0), got_children.pop(0), html, indent + 2))
    parts.append(' ' * indent)
    parts.append(self.collect_diff_end_tag(want, got))
    parts.append('\n')
    if strip(want.tail) or strip(got.tail):
        parts.append(' ' * indent)
        parts.append(self.collect_diff_text(want.tail, got.tail))
        parts.append('\n')
    return ''.join(parts)