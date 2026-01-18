import numpy as np
import holoviews as hv
from holoviews import opts
from holoviews.streams import RangeXY
from numba import jit
def get_fractal(x_range, y_range):
    (x0, x1), (y0, y1) = (x_range, y_range)
    image = np.zeros((600, 600), dtype=np.uint8)
    return hv.Image(create_fractal(x0, x1, -y1, -y0, image, 200), bounds=(x0, y0, x1, y1))