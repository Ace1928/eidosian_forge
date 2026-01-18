from html.entities import codepoint2name
from collections import defaultdict
import codecs
import re
import logging
import string
from html.entities import html5
@classmethod
def find_declared_encoding(cls, markup, is_html=False, search_entire_document=False):
    """Given a document, tries to find its declared encoding.

        An XML encoding is declared at the beginning of the document.

        An HTML encoding is declared in a <meta> tag, hopefully near the
        beginning of the document.

        :param markup: Some markup.
        :param is_html: If True, this markup is considered to be HTML. Otherwise
            it's assumed to be XML.
        :param search_entire_document: Since an encoding is supposed to declared near the beginning
            of the document, most of the time it's only necessary to search a few kilobytes of data.
            Set this to True to force this method to search the entire document.
        """
    if search_entire_document:
        xml_endpos = html_endpos = len(markup)
    else:
        xml_endpos = 1024
        html_endpos = max(2048, int(len(markup) * 0.05))
    if isinstance(markup, bytes):
        res = encoding_res[bytes]
    else:
        res = encoding_res[str]
    xml_re = res['xml']
    html_re = res['html']
    declared_encoding = None
    declared_encoding_match = xml_re.search(markup, endpos=xml_endpos)
    if not declared_encoding_match and is_html:
        declared_encoding_match = html_re.search(markup, endpos=html_endpos)
    if declared_encoding_match is not None:
        declared_encoding = declared_encoding_match.groups()[0]
    if declared_encoding:
        if isinstance(declared_encoding, bytes):
            declared_encoding = declared_encoding.decode('ascii', 'replace')
        return declared_encoding.lower()
    return None