import time
import warnings
from asgiref.sync import sync_to_async
from django.core.exceptions import ImproperlyConfigured
from django.utils.module_loading import import_string
from django.utils.regex_helper import _lazy_re_compile
def get_key_func(key_func):
    """
    Function to decide which key function to use.

    Default to ``default_key_func``.
    """
    if key_func is not None:
        if callable(key_func):
            return key_func
        else:
            return import_string(key_func)
    return default_key_func