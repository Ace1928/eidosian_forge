import re
import string
from .util import escape_url
def parse_link(src, pos):
    href, href_pos = parse_link_href(src, pos)
    if href is None:
        return (None, None)
    title, title_pos = parse_link_title(src, href_pos, len(src))
    next_pos = title_pos or href_pos
    m = PAREN_END_RE.match(src, next_pos)
    if not m:
        return (None, None)
    href = unescape_char(href)
    attrs = {'url': escape_url(href)}
    if title:
        attrs['title'] = title
    return (attrs, m.end())