import os
import pytest
from bs4 import (
def fuzz_test_with_css(self, filename):
    data = self.__markup(filename)
    parsers = ['lxml-xml', 'html5lib', 'html.parser', 'lxml']
    try:
        idx = int(data[0]) % len(parsers)
    except ValueError:
        return
    css_selector, data = (data[1:10], data[10:])
    try:
        soup = BeautifulSoup(data[1:], features=parsers[idx])
    except ParserRejectedMarkup:
        return
    except ValueError:
        return
    list(soup.find_all(True))
    try:
        soup.css.select(css_selector.decode('utf-8', 'replace'))
    except SelectorSyntaxError:
        return
    soup.prettify()