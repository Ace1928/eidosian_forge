import difflib
import json
import logging
import posixpath
import sys
import threading
import unittest
import warnings
from collections import Counter
from contextlib import contextmanager
from copy import copy, deepcopy
from difflib import get_close_matches
from functools import wraps
from unittest.suite import _DebugResult
from unittest.util import safe_repr
from urllib.parse import (
from urllib.request import url2pathname
from asgiref.sync import async_to_sync, iscoroutinefunction
from django.apps import apps
from django.conf import settings
from django.core import mail
from django.core.exceptions import ImproperlyConfigured, ValidationError
from django.core.files import locks
from django.core.handlers.wsgi import WSGIHandler, get_path_info
from django.core.management import call_command
from django.core.management.color import no_style
from django.core.management.sql import emit_post_migrate_signal
from django.core.servers.basehttp import ThreadedWSGIServer, WSGIRequestHandler
from django.core.signals import setting_changed
from django.db import DEFAULT_DB_ALIAS, connection, connections, transaction
from django.forms.fields import CharField
from django.http import QueryDict
from django.http.request import split_domain_port, validate_host
from django.test.client import AsyncClient, Client
from django.test.html import HTMLParseError, parse_html
from django.test.signals import template_rendered
from django.test.utils import (
from django.utils.deprecation import RemovedInDjango51Warning
from django.utils.functional import classproperty
from django.views.static import serve
def assertFormSetError(self, formset, form_index, field, errors, msg_prefix=''):
    """
        Similar to assertFormError() but for formsets.

        Use form_index=None to check the formset's non-form errors (in that
        case, you must also use field=None).
        Otherwise use an integer to check the formset's n-th form for errors.

        Other parameters are the same as assertFormError().
        """
    if form_index is None and field is not None:
        raise ValueError('You must use field=None with form_index=None.')
    if msg_prefix:
        msg_prefix += ': '
    errors = to_list(errors)
    if not formset.is_bound:
        self.fail(f'{msg_prefix}The formset {formset!r} is not bound, it will never have any errors.')
    if form_index is not None and form_index >= formset.total_form_count():
        form_count = formset.total_form_count()
        form_or_forms = 'forms' if form_count > 1 else 'form'
        self.fail(f'{msg_prefix}The formset {formset!r} only has {form_count} {form_or_forms}.')
    if form_index is not None:
        form_repr = f'form {form_index} of formset {formset!r}'
        self._assert_form_error(formset.forms[form_index], field, errors, msg_prefix, form_repr)
    else:
        failure_message = f"The non-form errors of formset {formset!r} don't match."
        self.assertEqual(formset.non_form_errors(), errors, msg_prefix + failure_message)