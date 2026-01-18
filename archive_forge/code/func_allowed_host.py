import sys
import unittest
from contextlib import contextmanager
from django.test import LiveServerTestCase, tag
from django.utils.functional import classproperty
from django.utils.module_loading import import_string
from django.utils.text import capfirst
@classproperty
def allowed_host(cls):
    return cls.external_host or cls.host