import sys
import unittest
import sys
def test_inside_exec(self):
    from zope.interface.advice import getFrameInfo
    _globals = {'getFrameInfo': getFrameInfo}
    _locals = {}
    exec(_FUNKY_EXEC, _globals, _locals)
    self.assertEqual(_locals['kind'], 'exec')
    self.assertTrue(_locals['f_locals'] is _locals)
    self.assertTrue(_locals['module'] is None)
    self.assertTrue(_locals['f_globals'] is _globals)