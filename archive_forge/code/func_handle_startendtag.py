import re
import _markupbase
from html import unescape
def handle_startendtag(self, tag, attrs):
    self.handle_starttag(tag, attrs)
    self.handle_endtag(tag)