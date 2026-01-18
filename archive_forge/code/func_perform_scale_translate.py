import copy
import re
import numpy as np
from plotly import colors
from ...core.util import isfinite, max_range
from ..util import color_intervals, process_cmap
def perform_scale_translate(obj):
    domain = obj.setdefault('domain', {})
    x = domain.get('x', [0, 1])
    y = domain.get('y', [0, 1])
    domain['x'] = scale_translate_x(x)
    domain['y'] = scale_translate_y(y)