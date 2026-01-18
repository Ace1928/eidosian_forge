import os
import sys
from textwrap import dedent
from twisted.persisted import sob
from twisted.persisted.styles import Ephemeral
from twisted.python import components
from twisted.trial import unittest
def testEverythingEphemeralException(self):
    """
        Test that an exception during load() won't cause _EE to mask __main__
        """
    dirname = self.mktemp()
    os.mkdir(dirname)
    filename = os.path.join(dirname, 'persisttest.ee_exception')
    with open(filename, 'w') as f:
        f.write('raise ValueError\n')
    self.assertRaises(ValueError, sob.load, filename, 'source')
    self.assertEqual(type(sys.modules['__main__']), FakeModule)