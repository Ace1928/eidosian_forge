from bs4 import BeautifulSoup, NavigableString, Comment, Doctype
from textwrap import fill
import re
import six
def should_convert_tag(self, tag):
    tag = tag.lower()
    strip = self.options['strip']
    convert = self.options['convert']
    if strip is not None:
        return tag not in strip
    elif convert is not None:
        return tag in convert
    else:
        return True