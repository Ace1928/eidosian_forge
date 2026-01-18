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
def assertNotContains(self, response, text, status_code=200, msg_prefix='', html=False):
    """
        Assert that a response indicates that some content was retrieved
        successfully, (i.e., the HTTP status code was as expected) and that
        ``text`` doesn't occur in the content of the response.
        """
    text_repr, real_count, msg_prefix = self._assert_contains(response, text, status_code, msg_prefix, html)
    self.assertEqual(real_count, 0, msg_prefix + 'Response should not contain %s' % text_repr)