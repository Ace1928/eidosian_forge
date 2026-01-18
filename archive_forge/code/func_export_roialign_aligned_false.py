import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
@staticmethod
def export_roialign_aligned_false() -> None:
    node = onnx.helper.make_node('RoiAlign', inputs=['X', 'rois', 'batch_indices'], outputs=['Y'], spatial_scale=1.0, output_height=5, output_width=5, sampling_ratio=2, coordinate_transformation_mode='output_half_pixel')
    X, batch_indices, rois = get_roi_align_input_values()
    Y = np.array([[[[0.4664, 0.4466, 0.3405, 0.5688, 0.6068], [0.3714, 0.4296, 0.3835, 0.5562, 0.351], [0.2768, 0.4883, 0.5222, 0.5528, 0.4171], [0.4713, 0.4844, 0.6904, 0.492, 0.8774], [0.6239, 0.7125, 0.6289, 0.3355, 0.3495]]], [[[0.3022, 0.4305, 0.4696, 0.3978, 0.5423], [0.3656, 0.705, 0.5165, 0.3172, 0.7015], [0.2912, 0.5059, 0.6476, 0.6235, 0.8299], [0.5916, 0.7389, 0.7048, 0.8372, 0.8893], [0.6227, 0.6153, 0.7097, 0.6154, 0.4585]]], [[[0.2384, 0.3379, 0.3717, 0.61, 0.7601], [0.3767, 0.3785, 0.7147, 0.9243, 0.9727], [0.5749, 0.5826, 0.5709, 0.7619, 0.877], [0.5355, 0.2566, 0.2141, 0.2796, 0.36], [0.4365, 0.3504, 0.2887, 0.3661, 0.2349]]]], dtype=np.float32)
    expect(node, inputs=[X, rois, batch_indices], outputs=[Y], name='test_roialign_aligned_false')