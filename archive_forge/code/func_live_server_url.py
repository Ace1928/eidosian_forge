import sys
import unittest
from contextlib import contextmanager
from django.test import LiveServerTestCase, tag
from django.utils.functional import classproperty
from django.utils.module_loading import import_string
from django.utils.text import capfirst
@classproperty
def live_server_url(cls):
    return 'http://%s:%s' % (cls.external_host or cls.host, cls.server_thread.port)