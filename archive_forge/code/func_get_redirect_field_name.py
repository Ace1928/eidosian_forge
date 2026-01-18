from urllib.parse import urlparse
from django.conf import settings
from django.contrib.auth import REDIRECT_FIELD_NAME
from django.contrib.auth.views import redirect_to_login
from django.core.exceptions import ImproperlyConfigured, PermissionDenied
from django.shortcuts import resolve_url
def get_redirect_field_name(self):
    """
        Override this method to override the redirect_field_name attribute.
        """
    return self.redirect_field_name