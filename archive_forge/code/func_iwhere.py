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
def iwhere(filename):
    possible_paths = _gen_possible_matches(filename)
    existing_file_paths = filter(os.path.isfile, possible_paths)
    return existing_file_paths