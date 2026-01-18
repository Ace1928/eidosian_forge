import numpy as np
import shutil
from ..util.dtype import img_as_bool
from ._registry import registry, registry_urls
from .. import __version__
import os.path as osp
import os
def lbp_frontal_face_cascade_filename():
    """Return the path to the XML file containing the weak classifier cascade.

    These classifiers were trained using LBP features. The file is part
    of the OpenCV repository [1]_.

    References
    ----------
    .. [1] OpenCV lbpcascade trained files
           https://github.com/opencv/opencv/tree/master/data/lbpcascades
    """
    return _fetch('data/lbpcascade_frontalface_opencv.xml')