import unittest
from traits.testing.optional_dependencies import (
import traits
@requires_pkg_resources
def test_dunder_version(self):
    self.assertIsInstance(traits.__version__, str)
    parsed_version = pkg_resources.parse_version(traits.__version__)
    self.assertEqual(str(parsed_version), traits.__version__)