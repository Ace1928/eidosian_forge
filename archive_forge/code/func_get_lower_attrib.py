import re
from w3lib.html import strip_html5_whitespace
from extruct.utils import parse_html
def get_lower_attrib(name):
    return re.sub('.*\\.', '', name).lower()