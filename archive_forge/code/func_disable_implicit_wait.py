import sys
import unittest
from contextlib import contextmanager
from django.test import LiveServerTestCase, tag
from django.utils.functional import classproperty
from django.utils.module_loading import import_string
from django.utils.text import capfirst
@contextmanager
def disable_implicit_wait(self):
    """Disable the default implicit wait."""
    self.selenium.implicitly_wait(0)
    try:
        yield
    finally:
        self.selenium.implicitly_wait(self.implicit_wait)