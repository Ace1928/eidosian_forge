from bs4 import BeautifulSoup, NavigableString, Comment, Doctype
from textwrap import fill
import re
import six
def convert_td(self, el, text, convert_as_inline):
    return ' ' + text + ' |'