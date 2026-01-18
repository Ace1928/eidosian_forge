import doctest
import os
import unittest
from genshi.core import Stream
from genshi.output import DocType
from genshi.template import MarkupTemplate, TextTemplate, NewTextTemplate
from genshi.template.plugin import ConfigurationError, \
def test_init_with_output_options(self):
    plugin = TextTemplateEnginePlugin(options={'genshi.default_encoding': 'iso-8859-15'})
    self.assertEqual('iso-8859-15', plugin.default_encoding)