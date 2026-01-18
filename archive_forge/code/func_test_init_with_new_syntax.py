import doctest
import os
import unittest
from genshi.core import Stream
from genshi.output import DocType
from genshi.template import MarkupTemplate, TextTemplate, NewTextTemplate
from genshi.template.plugin import ConfigurationError, \
def test_init_with_new_syntax(self):
    plugin = TextTemplateEnginePlugin(options={'genshi.new_text_syntax': 'yes'})
    self.assertEqual(NewTextTemplate, plugin.template_class)
    tmpl = plugin.load_template(PACKAGE + '.templates.new_syntax')
    output = plugin.render({'foo': True}, template=tmpl)
    self.assertEqual('bar', output)