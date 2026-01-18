import sys
import unittest
from contextlib import contextmanager
from django.test import LiveServerTestCase, tag
from django.utils.functional import classproperty
from django.utils.module_loading import import_string
from django.utils.text import capfirst
@classmethod
def import_options(cls, browser):
    return import_string('selenium.webdriver.%s.options.Options' % browser)