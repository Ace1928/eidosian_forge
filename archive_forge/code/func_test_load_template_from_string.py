import doctest
import os
import unittest
from genshi.core import Stream
from genshi.output import DocType
from genshi.template import MarkupTemplate, TextTemplate, NewTextTemplate
from genshi.template.plugin import ConfigurationError, \
def test_load_template_from_string(self):
    plugin = TextTemplateEnginePlugin()
    tmpl = plugin.load_template(None, template_string='$message')
    self.assertEqual(None, tmpl.filename)
    assert isinstance(tmpl, TextTemplate)