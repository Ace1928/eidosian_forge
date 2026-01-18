import copy
import re
from urllib.parse import urlsplit, unquote_plus
from lxml import etree
from lxml.html import defs
from lxml.html import fromstring, XHTML_NAMESPACE
from lxml.html import xhtml_to_html, _transform_result
def allow_element(self, el):
    """
        Decide whether an element is configured to be accepted or rejected.

        :param el: an element.
        :return: true to accept the element or false to reject/discard it.
        """
    if el.tag not in self._tag_link_attrs:
        return False
    attr = self._tag_link_attrs[el.tag]
    if isinstance(attr, (list, tuple)):
        for one_attr in attr:
            url = el.get(one_attr)
            if not url:
                return False
            if not self.allow_embedded_url(el, url):
                return False
        return True
    else:
        url = el.get(attr)
        if not url:
            return False
        return self.allow_embedded_url(el, url)