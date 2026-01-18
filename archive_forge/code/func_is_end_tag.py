import difflib
from lxml import etree
from lxml.html import fragment_fromstring
import re
def is_end_tag(tok):
    return tok.startswith('</')