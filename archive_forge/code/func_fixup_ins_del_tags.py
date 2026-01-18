import difflib
from lxml import etree
from lxml.html import fragment_fromstring
import re
def fixup_ins_del_tags(html):
    """ Given an html string, move any <ins> or <del> tags inside of any
    block-level elements, e.g. transform <ins><p>word</p></ins> to
    <p><ins>word</ins></p> """
    doc = parse_html(html, cleanup=False)
    _fixup_ins_del_tags(doc)
    html = serialize_html_fragment(doc, skip_outer=True)
    return html