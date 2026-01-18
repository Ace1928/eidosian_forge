import warnings
from typing import Tuple
from urllib.parse import urlparse
from w3lib.url import safe_url_string
from scrapy import signals
from scrapy.exceptions import NotConfigured
from scrapy.http import Request, Response
from scrapy.utils.misc import load_object
from scrapy.utils.python import to_unicode
from scrapy.utils.url import strip_url
def request_scheduled(self, request, spider):
    redirected_urls = request.meta.get('redirect_urls', [])
    if redirected_urls:
        request_referrer = request.headers.get('Referer')
        if request_referrer is not None:
            parent_url = safe_url_string(request_referrer)
            policy_referrer = self.policy(parent_url, request).referrer(parent_url, request.url)
            if policy_referrer != request_referrer:
                if policy_referrer is None:
                    request.headers.pop('Referer')
                else:
                    request.headers['Referer'] = policy_referrer