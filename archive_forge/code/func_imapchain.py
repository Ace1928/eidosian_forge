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
def imapchain(*a, **kwa):
    """ Like map but also chains the results. """
    imap_results = map(*a, **kwa)
    return itertools.chain(*imap_results)