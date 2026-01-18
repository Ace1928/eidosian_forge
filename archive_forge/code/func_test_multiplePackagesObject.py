import operator
import os
import shutil
import sys
import textwrap
import tempfile
from unittest import skipIf, TestCase
def test_multiplePackagesObject(self):
    """
        L{wrapFQPN} returns a L{twisted.python.modules.PythonAttribute}
        referring to the deepest object described by an FQPN,
        descending through several packages.
        """
    import xml.etree.ElementTree
    import automat
    for fqpn, obj in [('xml.etree.ElementTree.fromstring', xml.etree.ElementTree.fromstring), ('automat.MethodicalMachine.__doc__', automat.MethodicalMachine.__doc__)]:
        self.assertAttributeWrapperRefersTo(self.wrapFQPN(fqpn), fqpn, obj)