import operator
import os
import shutil
import sys
import textwrap
import tempfile
from unittest import skipIf, TestCase
def test_failsWithMissingSingleModuleOrPackage(self):
    """
        L{wrapFQPN} raises L{NoModule} when given a dotless FQPN that does
        not refer to a module or package.
        """
    with self.assertRaises(self.NoModule):
        self.wrapFQPN('this is not an acceptable name!')