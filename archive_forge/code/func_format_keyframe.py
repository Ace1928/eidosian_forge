import os
import json
import numpy as np
import ipywidgets as widgets
import pythreejs
import ipywebrtc
from IPython.display import display
def format_keyframe(self, time, p, q):
    x, y, z = p
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    lon = np.arctan2(y, x) * 180 / np.pi
    lat = (-np.arccos(z / r) + np.pi / 2) * 180 / np.pi
    return '{:.1f}s-r={:.2f}, {:.0f}/{:.0f}'.format(time, r, lon, lat)