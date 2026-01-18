import collections
import email.message
import functools
import itertools
import json
import logging
import os
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from optparse import Values
from typing import (
from pip._vendor import requests
from pip._vendor.requests import Response
from pip._vendor.requests.exceptions import RetryError, SSLError
from pip._internal.exceptions import NetworkConnectionError
from pip._internal.models.link import Link
from pip._internal.models.search_scope import SearchScope
from pip._internal.network.session import PipSession
from pip._internal.network.utils import raise_for_status
from pip._internal.utils.filetypes import is_archive_file
from pip._internal.utils.misc import redact_auth_from_url
from pip._internal.vcs import vcs
from .sources import CandidatesFromPage, LinkSource, build_source
@with_cached_index_content
def parse_links(page: 'IndexContent') -> Iterable[Link]:
    """
    Parse a Simple API's Index Content, and yield its anchor elements as Link objects.
    """
    content_type_l = page.content_type.lower()
    if content_type_l.startswith('application/vnd.pypi.simple.v1+json'):
        data = json.loads(page.content)
        for file in data.get('files', []):
            link = Link.from_json(file, page.url)
            if link is None:
                continue
            yield link
        return
    parser = HTMLLinkParser(page.url)
    encoding = page.encoding or 'utf-8'
    parser.feed(page.content.decode(encoding))
    url = page.url
    base_url = parser.base_url or url
    for anchor in parser.anchors:
        link = Link.from_element(anchor, page_url=url, base_url=base_url)
        if link is None:
            continue
        yield link