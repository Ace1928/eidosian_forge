import doctest
import os
import shutil
import tempfile
import unittest
from genshi.template.base import TemplateSyntaxError
from genshi.template.loader import TemplateLoader
from genshi.template.text import OldTextTemplate, NewTextTemplate
def test_empty_lines2(self):
    tmpl = NewTextTemplate('Your items:\n\n{% for item in items %}  * ${item}\n\n{% end %}')
    self.assertEqual('Your items:\n\n  * 0\n\n  * 1\n\n  * 2\n\n', tmpl.generate(items=range(3)).render(encoding=None))