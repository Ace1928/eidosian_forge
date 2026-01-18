import os
import sys
import tempfile
import textwrap
import shutil
import subprocess
import unittest
from traits.api import (
from traits.testing.optional_dependencies import requires_numpy
def test_method_deprecated(self):

    class A:

        def some_method(self):
            pass
    with self.assertWarnsRegex(DeprecationWarning, 'Method trait type'):
        Method()
    with self.assertWarnsRegex(DeprecationWarning, 'Method trait type'):
        Method(A().some_method, gluten_free=False)