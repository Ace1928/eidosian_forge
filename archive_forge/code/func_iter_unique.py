import os
import platform
import itertools
from .xmltodict import parse as xmltodictparser
import subprocess as sp
import numpy as np
from .edge import canny
from .stpyr import SpatialSteerablePyramid, rolling_window
from .mscn import compute_image_mscn_transform, gen_gauss_window
from .stats import ggd_features, aggd_features, paired_product
def iter_unique(iterable):
    yielded = set()
    for i in iterable:
        if i in yielded:
            continue
        yield i
        yielded.add(i)