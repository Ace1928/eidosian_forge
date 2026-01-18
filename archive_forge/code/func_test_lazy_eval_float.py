import unittest
from IPython.core.prompts import  LazyEvaluate
def test_lazy_eval_float(self):
    f = 0.503
    lz = LazyEvaluate(lambda: f)
    self.assertEqual(str(lz), str(f))
    self.assertEqual(format(lz), str(f))
    self.assertEqual(format(lz, '.1'), '0.5')