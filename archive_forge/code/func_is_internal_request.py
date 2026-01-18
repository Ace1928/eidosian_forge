import re
from urllib.parse import urlparse
from django.conf import settings
from django.core.exceptions import PermissionDenied
from django.core.mail import mail_managers
from django.http import HttpResponsePermanentRedirect
from django.urls import is_valid_path
from django.utils.deprecation import MiddlewareMixin
from django.utils.http import escape_leading_slashes
def is_internal_request(self, domain, referer):
    """
        Return True if the referring URL is the same domain as the current
        request.
        """
    return bool(re.match('^https?://%s/' % re.escape(domain), referer))