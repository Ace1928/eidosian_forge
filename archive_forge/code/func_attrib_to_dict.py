import re
from w3lib.html import strip_html5_whitespace
from extruct.utils import parse_html
def attrib_to_dict(attribs):
    return dict(attribs.items())