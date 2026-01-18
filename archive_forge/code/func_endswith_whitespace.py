from html.parser import HTMLParser
from itertools import zip_longest
def endswith_whitespace(self):
    return self._trailing_whitespace != '' or (self._stripped_data == '' and self._leading_whitespace != '')