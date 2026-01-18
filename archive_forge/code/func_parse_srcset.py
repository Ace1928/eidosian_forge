import re
from urllib.parse import urljoin
import bs4
from bs4.element import Comment, NavigableString, Tag
def parse_srcset(srcset, base_url):
    """Return a dictionary of sources found in srcset."""
    sources = {}
    for url, descriptor in re.findall('(\\S+)\\s*([\\d.]+[xw])?\\s*,?\\s*', srcset, re.MULTILINE):
        if not descriptor:
            descriptor = '1x'
        if descriptor not in sources:
            sources[descriptor] = try_urljoin(base_url, url.strip(','))
    return sources