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
def test_abbreviated_annotations(self):
    with self.create_directive() as directive:
        documenter = TraitDocumenter(directive, __name__ + '.MyTestClass.bar')
        documenter.generate(all_members=True)
        result = directive.result
    for item in result:
        if item.lstrip().startswith(':annotation:'):
            break
    else:
        self.fail("Didn't find the expected trait :annotation:")
    self.assertIn('First line', item)
    self.assertNotIn('\n', item)