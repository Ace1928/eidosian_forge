from lxml import etree
import sys
import re
import doctest
def collect_diff_end_tag(self, want, got):
    if want.tag != got.tag:
        tag = '%s (got: %s)' % (want.tag, got.tag)
    else:
        tag = got.tag
    return '</%s>' % tag