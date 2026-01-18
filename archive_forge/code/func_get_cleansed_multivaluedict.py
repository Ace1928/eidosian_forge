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
def get_cleansed_multivaluedict(self, request, multivaluedict):
    """
        Replace the keys in a MultiValueDict marked as sensitive with stars.
        This mitigates leaking sensitive POST parameters if something like
        request.POST['nonexistent_key'] throws an exception (#21098).
        """
    sensitive_post_parameters = getattr(request, 'sensitive_post_parameters', [])
    if self.is_active(request) and sensitive_post_parameters:
        multivaluedict = multivaluedict.copy()
        for param in sensitive_post_parameters:
            if param in multivaluedict:
                multivaluedict[param] = self.cleansed_substitute
    return multivaluedict