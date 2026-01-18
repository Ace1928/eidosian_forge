import re
from urllib.parse import urlparse
from django.conf import settings
from django.core.exceptions import PermissionDenied
from django.core.mail import mail_managers
from django.http import HttpResponsePermanentRedirect
from django.urls import is_valid_path
from django.utils.deprecation import MiddlewareMixin
from django.utils.http import escape_leading_slashes
def is_ignorable_request(self, request, uri, domain, referer):
    """
        Return True if the given request *shouldn't* notify the site managers
        according to project settings or in situations outlined by the inline
        comments.
        """
    if not referer:
        return True
    if settings.APPEND_SLASH and uri.endswith('/') and (referer == uri[:-1]):
        return True
    if not self.is_internal_request(domain, referer) and '?' in referer:
        return True
    parsed_referer = urlparse(referer)
    if parsed_referer.netloc in ['', domain] and parsed_referer.path == uri:
        return True
    return any((pattern.search(uri) for pattern in settings.IGNORABLE_404_URLS))