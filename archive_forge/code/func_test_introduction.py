import os
import unittest
from traits.examples._etsdemo_info import introduction
from traits.testing.optional_dependencies import requires_pkg_resources
@requires_pkg_resources
def test_introduction(self):
    response = introduction({})
    self.assertTrue(os.path.exists(response['root']))