import doctest
import os
import shutil
import tempfile
import unittest
from genshi.template.base import TemplateSyntaxError
from genshi.template.loader import TemplateLoader
from genshi.template.text import OldTextTemplate, NewTextTemplate
def test_empty_lines2_with_crlf(self):
    tmpl = NewTextTemplate('Your items:\r\n\r\n{% for item in items %}\\\r\n  * ${item}\r\n\r\n{% end %}')
    self.assertEqual('Your items:\r\n\r\n  * 0\r\n\r\n  * 1\r\n\r\n  * 2\r\n\r\n', tmpl.generate(items=range(3)).render(encoding=None))