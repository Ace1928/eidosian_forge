from lxml import etree
import sys
import re
import doctest
def compare_docs(self, want, got):
    if not self.tag_compare(want.tag, got.tag):
        return False
    if not self.text_compare(want.text, got.text, True):
        return False
    if not self.text_compare(want.tail, got.tail, True):
        return False
    if 'any' not in want.attrib:
        want_keys = sorted(want.attrib.keys())
        got_keys = sorted(got.attrib.keys())
        if want_keys != got_keys:
            return False
        for key in want_keys:
            if not self.text_compare(want.attrib[key], got.attrib[key], False):
                return False
    if want.text != '...' or len(want):
        want_children = list(want)
        got_children = list(got)
        while want_children or got_children:
            if not want_children or not got_children:
                return False
            want_first = want_children.pop(0)
            got_first = got_children.pop(0)
            if not self.compare_docs(want_first, got_first):
                return False
            if not got_children and want_first.tail == '...':
                break
    return True