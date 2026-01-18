from absl.testing import absltest
from absl.testing import parameterized
import ml_dtypes
import numpy as np
def assert_representable(val):
    self.assertEqual(make_val(val).item(), val)