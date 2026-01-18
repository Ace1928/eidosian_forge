import doctest
import os
import pickle
import shutil
import sys
import tempfile
import unittest
import six
from genshi.compat import BytesIO, StringIO
from genshi.core import Markup
from genshi.filters.i18n import Translator
from genshi.input import XML
from genshi.template.base import BadDirectiveError, TemplateSyntaxError
from genshi.template.loader import TemplateLoader, TemplateNotFound
from genshi.template.markup import MarkupTemplate
def test_exec_with_trailing_space(self):
    """
        Verify that a code block processing instruction with trailing space
        does not cause a syntax error (see ticket #127).
        """
    MarkupTemplate('<foo>\n          <?python\n            bar = 42\n          ?>\n        </foo>')