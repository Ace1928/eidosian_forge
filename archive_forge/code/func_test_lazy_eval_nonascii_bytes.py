import unittest
from IPython.core.prompts import  LazyEvaluate
def test_lazy_eval_nonascii_bytes(self):
    u = u'ünicødé'
    b = u.encode('utf8')
    lz = LazyEvaluate(lambda: b)
    self.assertEqual(str(lz), str(b))
    self.assertEqual(format(lz), str(b))