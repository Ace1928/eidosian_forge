import re
import textwrap
import param
from param.ipython import ParamPager
from param.parameterized import bothmethod
from .util import group_sanitizer, label_sanitizer
def inner_filter(name, p):
    name_match = re.search(pattern, name)
    if name_match is not None:
        return True
    if p.doc is not None:
        doc_match = re.search(pattern, p.doc)
        if doc_match is not None:
            return True
    return False