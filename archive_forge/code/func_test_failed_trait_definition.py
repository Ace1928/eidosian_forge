import contextlib
import io
import os
import shutil
import tempfile
import textwrap
import tokenize
import unittest
import unittest.mock as mock
from traits.api import Bool, HasTraits, Int, Property
from traits.testing.optional_dependencies import sphinx, requires_sphinx
def test_failed_trait_definition(self):
    with self.assertRaises(ValueError):
        trait_definition(cls=Fake, trait_name='not_a_trait')