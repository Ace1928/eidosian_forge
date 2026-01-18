from urllib.parse import urlparse
from django.conf import settings
from django.contrib.auth import REDIRECT_FIELD_NAME
from django.contrib.auth.views import redirect_to_login
from django.core.exceptions import ImproperlyConfigured, PermissionDenied
from django.shortcuts import resolve_url
def get_permission_required(self):
    """
        Override this method to override the permission_required attribute.
        Must return an iterable.
        """
    if self.permission_required is None:
        raise ImproperlyConfigured(f'{self.__class__.__name__} is missing the permission_required attribute. Define {self.__class__.__name__}.permission_required, or override {self.__class__.__name__}.get_permission_required().')
    if isinstance(self.permission_required, str):
        perms = (self.permission_required,)
    else:
        perms = self.permission_required
    return perms