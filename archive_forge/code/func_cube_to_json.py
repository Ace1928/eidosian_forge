from __future__ import division
import logging
import warnings
import math
from base64 import b64encode
import numpy as np
import PIL.Image
import ipywidgets
import ipywebrtc
from ipython_genutils.py3compat import string_types
from ipyvolume import utils
def cube_to_json(grid, obj=None):
    if grid is None or len(grid.shape) == 1:
        return None
    f = StringIO()
    image_shape, slice_shape, rows, columns, slices = cube_to_png(grid, obj.data_min, obj.data_max, f)
    image_url = 'data:image/png;base64,' + b64encode(f.getvalue()).decode('ascii')
    json = {'image_shape': image_shape, 'slice_shape': slice_shape, 'rows': rows, 'columns': columns, 'slices': slices, 'src': image_url}
    return json