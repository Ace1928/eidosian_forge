import doctest
import os
import unittest
from genshi.core import Stream
from genshi.output import DocType
from genshi.template import MarkupTemplate, TextTemplate, NewTextTemplate
from genshi.template.plugin import ConfigurationError, \
def test_init_with_invalid_doctype(self):
    self.assertRaises(ConfigurationError, MarkupTemplateEnginePlugin, options={'genshi.default_doctype': 'foobar'})