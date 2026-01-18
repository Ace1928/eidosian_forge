from typing import Generator, Tuple
from urllib.parse import urljoin
from scrapy.exceptions import NotSupported
from scrapy.http.common import obsolete_setter
from scrapy.http.headers import Headers
from scrapy.http.request import Request
from scrapy.link import Link
from scrapy.utils.trackref import object_ref
def css(self, *a, **kw):
    """Shortcut method implemented only by responses whose content
        is text (subclasses of TextResponse).
        """
    raise NotSupported("Response content isn't text")