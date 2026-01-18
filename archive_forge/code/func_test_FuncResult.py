import unittest
from pygame.threads import FuncResult, tmap, WorkerQueue, Empty, STOP
from pygame import threads, Surface, transform
import time
def test_FuncResult(self):
    """Ensure FuncResult sets its result and exception attributes"""
    fr = FuncResult(lambda x: x + 1)
    fr(2)
    self.assertEqual(fr.result, 3)
    self.assertIsNone(fr.exception, 'no exception should be raised')
    exception = ValueError('rast')

    def x(sdf):
        raise exception
    fr = FuncResult(x)
    fr(None)
    self.assertIs(fr.exception, exception)