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
def test_can_document_member(self):
    with self.create_directive() as directive:
        class_documenter = ClassDocumenter(directive, __name__ + '.FindTheTraits')
        class_documenter.parse_name()
        class_documenter.import_object()
        self.assertTrue(TraitDocumenter.can_document_member(INSTANCEATTR, 'an_int', True, class_documenter))
        self.assertTrue(TraitDocumenter.can_document_member(INSTANCEATTR, 'another_int', True, class_documenter))
        self.assertFalse(TraitDocumenter.can_document_member(INSTANCEATTR, 'magic_number', True, class_documenter))
        self.assertFalse(TraitDocumenter.can_document_member(INSTANCEATTR, 'not_a_trait', True, class_documenter))