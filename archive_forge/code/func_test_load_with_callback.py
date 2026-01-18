import doctest
import os
import shutil
import tempfile
import unittest
from genshi.core import TEXT
from genshi.template.loader import TemplateLoader
from genshi.template.markup import MarkupTemplate
def test_load_with_callback(self):
    fileobj = open(os.path.join(self.dirname, 'tmpl.html'), 'w')
    try:
        fileobj.write('<html>\n              <p>Hello</p>\n            </html>')
    finally:
        fileobj.close()

    def template_loaded(template):

        def my_filter(stream, ctxt):
            for kind, data, pos in stream:
                if kind is TEXT and data.strip():
                    data = ', '.join([data, data.lower()])
                yield (kind, data, pos)
        template.filters.insert(0, my_filter)
    loader = TemplateLoader([self.dirname], callback=template_loaded)
    tmpl = loader.load('tmpl.html')
    self.assertEqual('<html>\n              <p>Hello, hello</p>\n            </html>', tmpl.generate().render(encoding=None))
    tmpl = loader.load('tmpl.html')
    self.assertEqual('<html>\n              <p>Hello, hello</p>\n            </html>', tmpl.generate().render(encoding=None))