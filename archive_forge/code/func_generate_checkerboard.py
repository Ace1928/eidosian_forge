from __future__ import annotations
import io
import numpy as np
import onnx
from onnx.backend.test.case.base import Base
from onnx.backend.test.case.node import _image_decoder_data, expect
def generate_checkerboard(width: int, height: int, square_size: int) -> np.ndarray:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    num_squares_x = width // square_size
    num_squares_y = height // square_size
    colors = np.random.randint(0, 256, size=(num_squares_y, num_squares_x, 3), dtype=np.uint8)
    for i in range(num_squares_y):
        for j in range(num_squares_x):
            x = j * square_size
            y = i * square_size
            color = colors[i, j]
            image[y:y + square_size, x:x + square_size, :] = color
    return image