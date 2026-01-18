import os
import sys
from textwrap import dedent
from twisted.persisted import sob
from twisted.persisted.styles import Ephemeral
from twisted.python import components
from twisted.trial import unittest
def testEverythingEphemeralGetattr(self):
    """
        L{_EverythingEphermal.__getattr__} will proxy the __main__ module as an
        L{Ephemeral} object, and during load will be transparent, but after
        load will return L{Ephemeral} objects from any accessed attributes.
        """
    self.fakeMain.testMainModGetattr = 1
    dirname = self.mktemp()
    os.mkdir(dirname)
    filename = os.path.join(dirname, 'persisttest.ee_getattr')
    global mainWhileLoading
    mainWhileLoading = None
    with open(filename, 'w') as f:
        f.write(dedent('\n            app = []\n            import __main__\n            app.append(__main__.testMainModGetattr == 1)\n            try:\n                __main__.somethingElse\n            except AttributeError:\n                app.append(True)\n            else:\n                app.append(False)\n            from twisted.test import test_sob\n            test_sob.mainWhileLoading = __main__\n            '))
    loaded = sob.load(filename, 'source')
    self.assertIsInstance(loaded, list)
    self.assertTrue(loaded[0], 'Expected attribute not set.')
    self.assertTrue(loaded[1], 'Unexpected attribute set.')
    self.assertIsInstance(mainWhileLoading, Ephemeral)
    self.assertIsInstance(mainWhileLoading.somethingElse, Ephemeral)
    del mainWhileLoading