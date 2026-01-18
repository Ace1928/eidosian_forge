import copy
import re
from urllib.parse import urlsplit, unquote_plus
from lxml import etree
from lxml.html import defs
from lxml.html import fromstring, XHTML_NAMESPACE
from lxml.html import xhtml_to_html, _transform_result
def kill_conditional_comments(self, doc):
    """
        IE conditional comments basically embed HTML that the parser
        doesn't normally see.  We can't allow anything like that, so
        we'll kill any comments that could be conditional.
        """
    has_conditional_comment = _conditional_comment_re.search
    self._kill_elements(doc, lambda el: has_conditional_comment(el.text), etree.Comment)