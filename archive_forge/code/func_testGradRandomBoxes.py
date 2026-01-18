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
def testGradRandomBoxes(self):
    """Test that the gradient is correct for randomly generated boxes.

    The mapping is piecewise differentiable with respect to the box coordinates.
    The points where the function is not differentiable are those which are
    mapped to image pixels, i.e., the normalized y coordinates in
    np.linspace(0, 1, image_height) and normalized x coordinates in
    np.linspace(0, 1, image_width). Make sure that the box coordinates are
    sufficiently far away from those rectangular grid centers that are points of
    discontinuity, so that the finite difference Jacobian is close to the
    computed one.
    """
    np.random.seed(1)
    delta = 0.001
    radius = 2 * delta
    low, high = (-0.5, 1.5)
    image_height = 4
    for image_width in range(1, 3):
        for crop_height in range(1, 3):
            for crop_width in range(2, 4):
                for depth in range(1, 3):
                    for num_boxes in range(1, 3):
                        batch = num_boxes
                        image_shape = [batch, image_height, image_width, depth]
                        crop_size = [crop_height, crop_width]
                        image = np.arange(0, batch * image_height * image_width * depth).reshape(image_shape).astype(np.float32)
                        boxes = []
                        for _ in range(num_boxes):
                            y1, y2 = self._randomUniformAvoidAnchors(low, high, np.linspace(0, 1, image_height), radius, 2)
                            x1, x2 = self._randomUniformAvoidAnchors(low, high, np.linspace(0, 1, image_width), radius, 2)
                            boxes.append([y1, x1, y2, x2])
                        boxes = np.array(boxes, dtype=np.float32)
                        box_ind = np.arange(batch, dtype=np.int32)
                        image_tensor = constant_op.constant(image, shape=image_shape)
                        boxes_tensor = constant_op.constant(boxes, shape=[num_boxes, 4])
                        box_ind_tensor = constant_op.constant(box_ind, shape=[num_boxes])

                        def crop_resize(image_tensor, boxes_tensor):
                            return image_ops.crop_and_resize(image_tensor, boxes_tensor, box_ind_tensor, constant_op.constant(crop_size, shape=[2]))
                        with test_util.device(use_gpu=True):
                            with self.cached_session():
                                if config.is_op_determinism_enabled() and test_util.is_gpu_available():
                                    with self.assertRaises(errors_impl.UnimplementedError):
                                        gradient_checker_v2.compute_gradient(lambda x: crop_resize(x, boxes_tensor), [image_tensor])
                                    with self.assertRaises(errors_impl.UnimplementedError):
                                        gradient_checker_v2.compute_gradient(lambda x: crop_resize(image_tensor, x), [boxes_tensor])
                                else:
                                    err1 = gradient_checker_v2.max_error(*gradient_checker_v2.compute_gradient(lambda x: crop_resize(x, boxes_tensor), [image_tensor]))
                                    err2 = gradient_checker_v2.max_error(*gradient_checker_v2.compute_gradient(lambda x: crop_resize(image_tensor, x), [boxes_tensor]))
                                    err = max(err1, err2)
                                    self.assertLess(err, 0.002)