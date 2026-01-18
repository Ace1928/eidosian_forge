import sys
import unittest
import sys
def test_inside_function_call(self):
    from zope.interface.advice import getFrameInfo
    kind, module, f_locals, f_globals = getFrameInfo(sys._getframe())
    self.assertEqual(kind, 'function call')
    self.assertTrue(f_locals is locals())
    for d in (module.__dict__, f_globals):
        self.assertTrue(d is globals())