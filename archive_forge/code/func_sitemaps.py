import logging
import re
from collections import namedtuple
from datetime import time
from urllib.parse import ParseResult, quote, urlparse, urlunparse
@property
def sitemaps(self):
    """Get an iterator containing links to sitemaps specified."""
    return iter(self._sitemap_list)