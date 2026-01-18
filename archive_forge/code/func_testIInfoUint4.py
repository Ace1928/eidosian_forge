from absl.testing import absltest
from absl.testing import parameterized
import ml_dtypes
import numpy as np
def testIInfoUint4(self):
    info = ml_dtypes.iinfo(ml_dtypes.uint4)
    self.assertEqual(info.dtype, ml_dtypes.iinfo('uint4').dtype)
    self.assertEqual(info.dtype, ml_dtypes.iinfo(np.dtype('uint4')).dtype)
    self.assertEqual(info.min, 0)
    self.assertEqual(info.max, 15)
    self.assertEqual(info.dtype, np.dtype(ml_dtypes.uint4))
    self.assertEqual(info.bits, 4)
    self.assertEqual(info.kind, 'u')
    self.assertEqual(str(info), 'iinfo(min=0, max=15, dtype=uint4)')