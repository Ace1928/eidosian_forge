import sys
import unittest
from contextlib import contextmanager
from django.test import LiveServerTestCase, tag
from django.utils.functional import classproperty
from django.utils.module_loading import import_string
from django.utils.text import capfirst
@classmethod
def get_capability(cls, browser):
    from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
    return getattr(DesiredCapabilities, browser.upper())