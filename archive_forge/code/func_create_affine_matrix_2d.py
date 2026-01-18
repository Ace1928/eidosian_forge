import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import expect
from onnx.reference.ops.op_affine_grid import (
def create_affine_matrix_2d(angle1, offset_x, offset_y, shear_x, shear_y, scale_x, scale_y):
    rot = np.stack([np.cos(angle1), -np.sin(angle1), np.sin(angle1), np.cos(angle1)], axis=-1).reshape(-1, 2, 2)
    shear = np.stack([np.ones_like(shear_x), shear_x, shear_y, np.ones_like(shear_x)], axis=-1).reshape(-1, 2, 2)
    scale = np.stack([scale_x, np.zeros_like(scale_x), np.zeros_like(scale_x), scale_y], axis=-1).reshape(-1, 2, 2)
    translation = np.transpose(np.array([offset_x, offset_y])).reshape(-1, 1, 2)
    rotation_matrix = rot @ shear @ scale
    rotation_matrix = np.transpose(rotation_matrix, (0, 2, 1))
    affine_matrix = np.hstack((rotation_matrix, translation))
    affine_matrix = np.transpose(affine_matrix, (0, 2, 1))
    return affine_matrix.astype(np.float32)