import time
import warnings
from asgiref.sync import sync_to_async
from django.core.exceptions import ImproperlyConfigured
from django.utils.module_loading import import_string
from django.utils.regex_helper import _lazy_re_compile
def make_and_validate_key(self, key, version=None):
    """Helper to make and validate keys."""
    key = self.make_key(key, version=version)
    self.validate_key(key)
    return key