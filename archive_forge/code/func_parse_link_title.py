import re
import string
from .util import escape_url
def parse_link_title(src, start_pos, max_pos):
    m = LINK_TITLE_RE.match(src, start_pos, max_pos)
    if m:
        title = m.group(1)[1:-1]
        title = unescape_char(title)
        return (title, m.end())
    return (None, None)