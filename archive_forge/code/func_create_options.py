import sys
import unittest
from contextlib import contextmanager
from django.test import LiveServerTestCase, tag
from django.utils.functional import classproperty
from django.utils.module_loading import import_string
from django.utils.text import capfirst
def create_options(self):
    options = self.import_options(self.browser)()
    if self.headless:
        match self.browser:
            case 'chrome':
                options.add_argument('--headless=new')
            case 'firefox':
                options.add_argument('-headless')
    return options