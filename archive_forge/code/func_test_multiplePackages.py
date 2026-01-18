import operator
import os
import shutil
import sys
import textwrap
import tempfile
from unittest import skipIf, TestCase
def test_multiplePackages(self):
    """
        L{wrapFQPN} returns a L{twisted.python.modules.PythonModule}
        referring to the deepest package described by dotted FQPN.
        """
    import xml.etree
    self.assertModuleWrapperRefersTo(self.wrapFQPN('xml.etree'), xml.etree)