from __future__ import annotations
import logging
import os
import urllib.parse
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from functools import wraps
import idna
import requests
from .cache import DiskCache, get_cache_dir
from .remote import lenient_netloc, looks_like_ip, looks_like_ipv6
from .suffix_list import get_suffix_lists
def extract_urllib(self, url: urllib.parse.ParseResult | urllib.parse.SplitResult, include_psl_private_domains: bool | None=None, session: requests.Session | None=None) -> ExtractResult:
    """Take the output of urllib.parse URL parsing methods and further splits the parsed URL.

        Splits the parsed URL into its subdomain, domain, and suffix
        components, i.e. its effective TLD, gTLD, ccTLD, etc. components.

        This method is like `extract_str` but faster, as the string's domain
        name has already been parsed.

        >>> extractor = TLDExtract()
        >>> extractor.extract_urllib(urllib.parse.urlsplit('http://forums.news.cnn.com/'))
        ExtractResult(subdomain='forums.news', domain='cnn', suffix='com', is_private=False)
        >>> extractor.extract_urllib(urllib.parse.urlsplit('http://forums.bbc.co.uk/'))
        ExtractResult(subdomain='forums', domain='bbc', suffix='co.uk', is_private=False)
        """
    return self._extract_netloc(url.netloc, include_psl_private_domains, session=session)