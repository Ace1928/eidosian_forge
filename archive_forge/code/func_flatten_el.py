import difflib
from lxml import etree
from lxml.html import fragment_fromstring
import re
def flatten_el(el, include_hrefs, skip_tag=False):
    """ Takes an lxml element el, and generates all the text chunks for
    that tag.  Each start tag is a chunk, each word is a chunk, and each
    end tag is a chunk.

    If skip_tag is true, then the outermost container tag is
    not returned (just its contents)."""
    if not skip_tag:
        if el.tag == 'img':
            yield ('img', el.get('src'), start_tag(el))
        else:
            yield start_tag(el)
    if el.tag in empty_tags and (not el.text) and (not len(el)) and (not el.tail):
        return
    start_words = split_words(el.text)
    for word in start_words:
        yield html_escape(word)
    for child in el:
        yield from flatten_el(child, include_hrefs=include_hrefs)
    if el.tag == 'a' and el.get('href') and include_hrefs:
        yield ('href', el.get('href'))
    if not skip_tag:
        yield end_tag(el)
        end_words = split_words(el.tail)
        for word in end_words:
            yield html_escape(word)