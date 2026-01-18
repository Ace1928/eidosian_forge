import doctest
import os
import shutil
import tempfile
import unittest
from genshi.template.base import TemplateSyntaxError
from genshi.template.loader import TemplateLoader
from genshi.template.text import OldTextTemplate, NewTextTemplate
def test_include_expr(self):
    file1 = open(os.path.join(self.dirname, 'tmpl1.txt'), 'wb')
    try:
        file1.write(u'Included'.encode('utf-8'))
    finally:
        file1.close()
    file2 = open(os.path.join(self.dirname, 'tmpl2.txt'), 'wb')
    try:
        file2.write(u"----- Included data below this line -----\n    {% include ${'%s.txt' % ('tmpl1',)} %}\n    ----- Included data above this line -----".encode('utf-8'))
    finally:
        file2.close()
    loader = TemplateLoader([self.dirname])
    tmpl = loader.load('tmpl2.txt', cls=NewTextTemplate)
    self.assertEqual('----- Included data below this line -----\n    Included\n    ----- Included data above this line -----', tmpl.generate().render(encoding=None))