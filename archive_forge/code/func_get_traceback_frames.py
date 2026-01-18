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
def get_traceback_frames(self):
    exceptions = []
    exc_value = self.exc_value
    while exc_value:
        exceptions.append(exc_value)
        exc_value = self._get_explicit_or_implicit_cause(exc_value)
        if exc_value in exceptions:
            warnings.warn("Cycle in the exception chain detected: exception '%s' encountered again." % exc_value, ExceptionCycleWarning)
            break
    frames = []
    if not exceptions:
        return frames
    exc_value = exceptions.pop()
    tb = self.tb if not exceptions else exc_value.__traceback__
    while True:
        frames.extend(self.get_exception_traceback_frames(exc_value, tb))
        try:
            exc_value = exceptions.pop()
        except IndexError:
            break
        tb = exc_value.__traceback__
    return frames