import logging
import operator
from functools import partial
from urllib.parse import urljoin, urlparse
from lxml import etree
from parsel.csstranslator import HTMLTranslator
from w3lib.html import strip_html5_whitespace
from w3lib.url import canonicalize_url, safe_url_string
from scrapy.link import Link
from scrapy.linkextractors import (
from scrapy.utils.misc import arg_to_iter, rel_has_nofollow
from scrapy.utils.python import unique as unique_list
from scrapy.utils.response import get_base_url
from scrapy.utils.url import url_has_any_extension, url_is_from_any_domain
def extract_links(self, response):
    """Returns a list of :class:`~scrapy.link.Link` objects from the
        specified :class:`response <scrapy.http.Response>`.

        Only links that match the settings passed to the ``__init__`` method of
        the link extractor are returned.

        Duplicate links are omitted if the ``unique`` attribute is set to ``True``,
        otherwise they are returned.
        """
    base_url = get_base_url(response)
    if self.restrict_xpaths:
        docs = [subdoc for x in self.restrict_xpaths for subdoc in response.xpath(x)]
    else:
        docs = [response.selector]
    all_links = []
    for doc in docs:
        links = self._extract_links(doc, response.url, response.encoding, base_url)
        all_links.extend(self._process_links(links))
    if self.link_extractor.unique:
        return unique_list(all_links)
    return all_links