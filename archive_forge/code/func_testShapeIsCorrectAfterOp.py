from absl.testing import parameterized
import numpy as np
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.ops import gradient_checker_v2
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
def testShapeIsCorrectAfterOp(self):
    in_shape = [2, 20, 30, 3]
    out_shape = [2, 20, 30, 3]
    for nptype in self.TYPES:
        x = np.random.randint(0, high=255, size=[2, 20, 30, 3]).astype(nptype)
        rgb_input_tensor = constant_op.constant(x, shape=in_shape)
        hsv_out = gen_image_ops.rgb_to_hsv(rgb_input_tensor)
        with self.cached_session():
            self.assertEqual(out_shape, list(hsv_out.get_shape()))
        hsv_out = self.evaluate(hsv_out)
        self.assertEqual(out_shape, list(hsv_out.shape))