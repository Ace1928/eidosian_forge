import collections
import multiprocessing
import os
import threading
import warnings
import numpy as np
from keras.src import backend
from keras.src.api_export import keras_export
from keras.src.trainers.data_adapters.py_dataset_adapter import PyDataset
from keras.src.utils import image_utils
from keras.src.utils import io_utils
from keras.src.utils.module_utils import scipy
def get_random_transform(self, img_shape, seed=None):
    """Generates random parameters for a transformation.

        Args:
            img_shape: Tuple of integers.
                Shape of the image that is transformed.
            seed: Random seed.

        Returns:
            A dictionary containing randomly chosen parameters describing the
            transformation.
        """
    img_row_axis = self.row_axis - 1
    img_col_axis = self.col_axis - 1
    if seed is not None:
        np.random.seed(seed)
    if self.rotation_range:
        theta = np.random.uniform(-self.rotation_range, self.rotation_range)
    else:
        theta = 0
    if self.height_shift_range:
        try:
            tx = np.random.choice(self.height_shift_range)
            tx *= np.random.choice([-1, 1])
        except ValueError:
            tx = np.random.uniform(-self.height_shift_range, self.height_shift_range)
        if np.max(self.height_shift_range) < 1:
            tx *= img_shape[img_row_axis]
    else:
        tx = 0
    if self.width_shift_range:
        try:
            ty = np.random.choice(self.width_shift_range)
            ty *= np.random.choice([-1, 1])
        except ValueError:
            ty = np.random.uniform(-self.width_shift_range, self.width_shift_range)
        if np.max(self.width_shift_range) < 1:
            ty *= img_shape[img_col_axis]
    else:
        ty = 0
    if self.shear_range:
        shear = np.random.uniform(-self.shear_range, self.shear_range)
    else:
        shear = 0
    if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
        zx, zy = (1, 1)
    else:
        zx, zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)
    flip_horizontal = (np.random.random() < 0.5) * self.horizontal_flip
    flip_vertical = (np.random.random() < 0.5) * self.vertical_flip
    channel_shift_intensity = None
    if self.channel_shift_range != 0:
        channel_shift_intensity = np.random.uniform(-self.channel_shift_range, self.channel_shift_range)
    brightness = None
    if self.brightness_range is not None:
        brightness = np.random.uniform(self.brightness_range[0], self.brightness_range[1])
    transform_parameters = {'theta': theta, 'tx': tx, 'ty': ty, 'shear': shear, 'zx': zx, 'zy': zy, 'flip_horizontal': flip_horizontal, 'flip_vertical': flip_vertical, 'channel_shift_intensity': channel_shift_intensity, 'brightness': brightness}
    return transform_parameters