import functools
import inspect
import itertools
import re
import sys
import types
import warnings
from pathlib import Path
from django.conf import settings
from django.http import Http404, HttpResponse, HttpResponseNotFound
from django.template import Context, Engine, TemplateDoesNotExist
from django.template.defaultfilters import pprint
from django.urls import resolve
from django.utils import timezone
from django.utils.datastructures import MultiValueDict
from django.utils.encoding import force_str
from django.utils.module_loading import import_string
from django.utils.regex_helper import _lazy_re_compile
from django.utils.version import PY311, get_docs_version
from django.views.decorators.debug import coroutine_functions_to_sensitive_variables
def cleanse_setting(self, key, value):
    """
        Cleanse an individual setting key/value of sensitive content. If the
        value is a dictionary, recursively cleanse the keys in that dictionary.
        """
    if key == settings.SESSION_COOKIE_NAME:
        is_sensitive = True
    else:
        try:
            is_sensitive = self.hidden_settings.search(key)
        except TypeError:
            is_sensitive = False
    if is_sensitive:
        cleansed = self.cleansed_substitute
    elif isinstance(value, dict):
        cleansed = {k: self.cleanse_setting(k, v) for k, v in value.items()}
    elif isinstance(value, list):
        cleansed = [self.cleanse_setting('', v) for v in value]
    elif isinstance(value, tuple):
        cleansed = tuple([self.cleanse_setting('', v) for v in value])
    else:
        cleansed = value
    if callable(cleansed):
        cleansed = CallableSettingWrapper(cleansed)
    return cleansed