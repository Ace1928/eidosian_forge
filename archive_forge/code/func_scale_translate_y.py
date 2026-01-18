import copy
import re
import numpy as np
from plotly import colors
from ...core.util import isfinite, max_range
from ..util import color_intervals, process_cmap
def scale_translate_y(y):
    return [min(y[0] * scale_y + translate_y, 1), min(y[1] * scale_y + translate_y, 1)]