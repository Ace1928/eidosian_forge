import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_roialign_aligned_true() -> None:
    node = onnx.helper.make_node('RoiAlign', inputs=['X', 'rois', 'batch_indices'], outputs=['Y'], spatial_scale=1.0, output_height=5, output_width=5, sampling_ratio=2, coordinate_transformation_mode='half_pixel')
    X, batch_indices, rois = get_roi_align_input_values()
    Y = np.array([[[[0.5178, 0.3434, 0.3229, 0.4474, 0.6344], [0.4031, 0.5366, 0.4428, 0.4861, 0.4023], [0.2512, 0.4002, 0.5155, 0.6954, 0.3465], [0.335, 0.4601, 0.5881, 0.3439, 0.6849], [0.4932, 0.7141, 0.8217, 0.4719, 0.4039]]], [[[0.307, 0.2187, 0.3337, 0.488, 0.487], [0.1871, 0.4914, 0.5561, 0.4192, 0.3686], [0.1433, 0.4608, 0.5971, 0.531, 0.4982], [0.2788, 0.4386, 0.6022, 0.7, 0.7524], [0.5774, 0.7024, 0.7251, 0.7338, 0.8163]]], [[[0.2393, 0.4075, 0.3379, 0.2525, 0.4743], [0.3671, 0.2702, 0.4105, 0.6419, 0.8308], [0.5556, 0.4543, 0.5564, 0.7502, 0.93], [0.6626, 0.5617, 0.4813, 0.4954, 0.6663], [0.6636, 0.3721, 0.2056, 0.1928, 0.2478]]]], dtype=np.float32)
    expect(node, inputs=[X, rois, batch_indices], outputs=[Y], name='test_roialign_aligned_true')