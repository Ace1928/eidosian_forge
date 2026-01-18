import doctest
import os
import unittest
from genshi.core import Stream
from genshi.output import DocType
from genshi.template import MarkupTemplate, TextTemplate, NewTextTemplate
from genshi.template.plugin import ConfigurationError, \
def test_init_with_loader_options(self):
    plugin = TextTemplateEnginePlugin(options={'genshi.auto_reload': 'off', 'genshi.max_cache_size': '100', 'genshi.search_path': '/usr/share/tmpl:/usr/local/share/tmpl'})
    self.assertEqual(['/usr/share/tmpl', '/usr/local/share/tmpl'], plugin.loader.search_path)
    self.assertEqual(False, plugin.loader.auto_reload)
    self.assertEqual(100, plugin.loader._cache.capacity)