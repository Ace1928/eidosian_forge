from functools import partial
from django.contrib import auth
from django.contrib.auth import load_backend
from django.contrib.auth.backends import RemoteUserBackend
from django.core.exceptions import ImproperlyConfigured
from django.utils.deprecation import MiddlewareMixin
from django.utils.functional import SimpleLazyObject
def clean_username(self, username, request):
    """
        Allow the backend to clean the username, if the backend defines a
        clean_username method.
        """
    backend_str = request.session[auth.BACKEND_SESSION_KEY]
    backend = auth.load_backend(backend_str)
    try:
        username = backend.clean_username(username)
    except AttributeError:
        pass
    return username