from lxml import etree
import sys
import re
import doctest
def norm_whitespace(v):
    return _norm_whitespace_re.sub(' ', v)