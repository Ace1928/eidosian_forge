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
def cleanse_special_types(self, request, value):
    try:
        is_multivalue_dict = isinstance(value, MultiValueDict)
    except Exception as e:
        return '{!r} while evaluating {!r}'.format(e, value)
    if is_multivalue_dict:
        value = self.get_cleansed_multivaluedict(request, value)
    return value