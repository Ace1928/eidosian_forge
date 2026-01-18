import re
from formencode.rewritingparser import RewritingParser, html_quote
def insert_at_marker(self, marker, text):
    for i, item in enumerate(self._content):
        if item == (marker,):
            self._content.insert(i, text)
            break
    else:
        self._content.insert(0, text)