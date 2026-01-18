import doctest
import os
import shutil
import tempfile
import unittest
from genshi.template.base import TemplateSyntaxError
from genshi.template.loader import TemplateLoader
from genshi.template.text import OldTextTemplate, NewTextTemplate
def test_comment_escaping(self):
    tmpl = NewTextTemplate('\\{# escaped comment #}')
    self.assertEqual('{# escaped comment #}', tmpl.generate().render(encoding=None))