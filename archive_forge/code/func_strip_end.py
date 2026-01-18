import re
from urllib.parse import quote
from html import _replace_charref
def strip_end(src: str):
    return _strip_end_re.sub('\n', src)