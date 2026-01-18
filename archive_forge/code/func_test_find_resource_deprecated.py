import os
import tempfile
import unittest
from traits.util.resource import find_resource, store_resource
def test_find_resource_deprecated(self):
    with self.assertWarns(DeprecationWarning):
        find_resource('traits', os.path.join('traits', '__init__.py'))