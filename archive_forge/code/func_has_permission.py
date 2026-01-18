from urllib.parse import urlparse
from django.conf import settings
from django.contrib.auth import REDIRECT_FIELD_NAME
from django.contrib.auth.views import redirect_to_login
from django.core.exceptions import ImproperlyConfigured, PermissionDenied
from django.shortcuts import resolve_url
def has_permission(self):
    """
        Override this method to customize the way permissions are checked.
        """
    perms = self.get_permission_required()
    return self.request.user.has_perms(perms)